"""
Merge LoRA adapter weights into the base model and save a standalone model.

After merging, the resulting directory can be used with any standard inference
tool (transformers, vLLM, web_demo_mm.py, etc.) without requiring the PEFT
library at inference time.

Usage
-----
python tools/merge_lora.py \
    --model_path /path/to/Qwen3-VL-2B-Instruct \
    --checkpoint_path ./checkpoints/checkpoint-500 \
    --output_path ./merged_model

# Verify the merged model with a quick inference:
python tools/inference.py \
    --model_path ./merged_model \
    --prompt "Hello, describe yourself."
"""

import argparse
import os
import re
import torch
from pathlib import Path
from typing import Union


def _detect_model_type(model_path: str) -> str:
    """Detect the Qwen-VL model family from the model config or path name.

    Detection order:
    1. Read ``config.json`` from a local directory (most reliable; works for
       merged / renamed model directories whose path no longer contains the
       original model name).
    2. Fall back to path/name string matching (useful for HuggingFace Hub IDs
       where no local config file exists yet).

    MoE models (e.g. Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct)
    are identified either by ``model_type == "qwen3_vl_moe"`` in the config or
    by the ``<total>B-A<active>B`` naming pattern in the path.
    """
    import json

    # --- 1. Config-file based detection (local directories) -----------------
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            config_model_type = cfg.get("model_type", "").lower()
            if config_model_type == "qwen3_vl_moe":
                return "qwen3vl_moe"
            elif config_model_type == "qwen3_vl":
                # Still check for MoE via path pattern in case the config
                # uses a shared model_type for both dense and MoE variants.
                name = Path(model_path.rstrip("/")).name.lower()
                if re.search(r"\d+b-a\d+b", name):
                    return "qwen3vl_moe"
                return "qwen3vl"
            elif config_model_type == "qwen2_5_vl":
                return "qwen2.5vl"
            elif config_model_type == "qwen2_vl":
                return "qwen2vl"
            # Unknown config model_type — fall through to path-name heuristics
        except (OSError, json.JSONDecodeError):
            pass  # Fall through to path-name heuristics

    # --- 2. Path/name string matching (HuggingFace Hub IDs, etc.) -----------
    name = Path(model_path.rstrip("/")).name.lower()
    full = model_path.lower()
    # MoE pattern: something like "30b-a3b" or "235b-a22b"
    is_moe = bool(re.search(r"\d+b-a\d+b", name))
    if "qwen3" in full and is_moe:
        return "qwen3vl_moe"
    elif "qwen3" in full:
        return "qwen3vl"
    elif "qwen2.5" in full:
        return "qwen2.5vl"
    else:
        return "qwen2vl"


def load_base_model(model_path: str, torch_dtype: Union[str, torch.dtype]):
    model_type = _detect_model_type(model_path)

    if model_type == "qwen3vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path, dtype=torch_dtype, device_map="cpu"
        )
    elif model_type == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch_dtype, device_map="cpu"
        )
    elif model_type == "qwen2.5vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch_dtype, device_map="cpu"
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch_dtype, device_map="cpu"
        )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter checkpoint into the base model."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the base pretrained model (HuggingFace ID or local directory).",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help=(
            "Path to the LoRA checkpoint directory "
            "(e.g. ./checkpoints/checkpoint-500). "
            "Must contain adapter_config.json."
        ),
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory where the merged model will be saved.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype to use when saving the merged model (default: bfloat16).",
    )
    args = parser.parse_args()

    # Validate that the checkpoint contains a LoRA adapter config
    adapter_cfg = os.path.join(args.checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        raise FileNotFoundError(
            f"adapter_config.json not found in {args.checkpoint_path}. "
            "Make sure --checkpoint_path points to a LoRA checkpoint directory."
        )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    print(f"Loading base model from {args.model_path} ...")
    model = load_base_model(args.model_path, torch_dtype)

    print(f"Loading LoRA adapter from {args.checkpoint_path} ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.checkpoint_path)

    print("Merging LoRA weights into the base model ...")
    model = model.merge_and_unload()

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving merged model to {args.output_path} ...")
    model.save_pretrained(args.output_path)

    # Save processor / tokenizer so the merged directory is self-contained
    from transformers import AutoProcessor
    # Prefer using the processor saved alongside the adapter; fall back to base model
    processor_src = args.checkpoint_path
    if not os.path.exists(os.path.join(processor_src, "preprocessor_config.json")):
        processor_src = args.model_path
    print(f"Saving processor from {processor_src} ...")
    processor = AutoProcessor.from_pretrained(processor_src)
    processor.save_pretrained(args.output_path)

    print(
        f"\nDone! Merged model saved to: {args.output_path}\n"
        "You can now use it with:\n"
        f"  python tools/inference.py --model_path {args.output_path} --prompt '...'"
    )


if __name__ == "__main__":
    main()
