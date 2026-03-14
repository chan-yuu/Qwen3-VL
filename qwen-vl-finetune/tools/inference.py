"""
Inference script for fine-tuned Qwen-VL models.

Supports:
  - LoRA checkpoints (adapter weights loaded on top of the base model)
  - Fully fine-tuned / merged checkpoints (standard model directory)
  - Qwen2-VL, Qwen2.5-VL, Qwen3-VL (dense), Qwen3-VL (MoE) model families
  - Image and video inputs

Usage examples
--------------
# Run with a LoRA checkpoint (adapter merged at inference time):
python tools/inference.py \
    --model_path /path/to/Qwen3-VL-2B-Instruct \
    --checkpoint_path ./checkpoints/checkpoint-500 \
    --image /path/to/image.jpg \
    --prompt "Describe this image."

# Run with a fully fine-tuned / merged model:
python tools/inference.py \
    --model_path ./checkpoints \
    --image /path/to/image.jpg \
    --prompt "Describe this image."

# Run on a video:
python tools/inference.py \
    --model_path /path/to/Qwen3-VL-2B-Instruct \
    --checkpoint_path ./checkpoints/checkpoint-500 \
    --video /path/to/video.mp4 \
    --prompt "Describe what happens in this video."

# Text-only query:
python tools/inference.py \
    --model_path /path/to/Qwen3-VL-2B-Instruct \
    --checkpoint_path ./checkpoints/checkpoint-500 \
    --prompt "What is the capital of France?"
"""

import argparse
import os
import re
import sys
import torch
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Path setup — ensure repo-local packages are importable even when the script
# is invoked directly (i.e. without a prior `pip install`).
#
# Repository layout:
#   <repo_root>/
#     qwen-vl-finetune/
#       tools/
#         inference.py   ← this file
#     qwen-vl-utils/
#       src/
#         qwen_vl_utils/ ← package used at inference time
# ---------------------------------------------------------------------------
_TOOLS_DIR = Path(__file__).resolve().parent          # …/qwen-vl-finetune/tools
_FINETUNE_DIR = _TOOLS_DIR.parent                     # …/qwen-vl-finetune
_REPO_ROOT = _FINETUNE_DIR.parent                     # repo root
_QWEN_VL_UTILS_SRC = _REPO_ROOT / "qwen-vl-utils" / "src"

for _p in [str(_REPO_ROOT), str(_QWEN_VL_UTILS_SRC), str(_FINETUNE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_model_type(model_path: str) -> str:
    """Detect the Qwen-VL model family from the model path or name.

    MoE models (e.g. Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct)
    are identified by the ``<total>B-A<active>B`` naming pattern.
    """
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


def load_model(
    model_path: str,
    checkpoint_path: Optional[str],
    device: str,
    torch_dtype: Union[str, torch.dtype],
):
    """Load a base model and, when provided, attach a LoRA adapter checkpoint."""
    model_type = _detect_model_type(model_path)

    if model_type == "qwen3vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device
        )
    elif model_type == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device
        )
    elif model_type == "qwen2.5vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device
        )

    if checkpoint_path:
        # Detect whether the checkpoint directory contains a LoRA adapter config.
        adapter_cfg = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            from peft import PeftModel
            print(f"Loading LoRA adapter from {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path)
        else:
            print(
                f"Warning: {checkpoint_path} does not contain adapter_config.json. "
                "Assuming it is a full fine-tuned model directory and ignoring --checkpoint_path. "
                "Pass the directory as --model_path instead."
            )

    model.eval()
    return model, model_type


def build_messages(
    prompt: str,
    image: Optional[str],
    video: Optional[str],
) -> list:
    content = []
    if image:
        content.append({"type": "image", "image": image})
    if video:
        content.append({"type": "video", "video": video})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_inference(
    model,
    processor,
    model_type: str,
    messages: list,
    max_new_tokens: int,
    enable_thinking: bool,
):
    from qwen_vl_utils import process_vision_info

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if model_type in ("qwen3vl", "qwen3vl_moe"):
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
    else:
        if model_type == "qwen2.5vl":
            images, videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
        else:
            images, videos = process_vision_info(messages)
            video_kwargs = {}

        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generate_kwargs = dict(max_new_tokens=max_new_tokens)
    if model_type in ("qwen3vl", "qwen3vl_moe"):
        generate_kwargs["enable_thinking"] = enable_thinking

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    # Trim the input tokens from the output
    input_length = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_length:]
    response = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned (or LoRA) Qwen-VL model."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help=(
            "Path to the base model (HuggingFace model ID or local directory). "
            "When using LoRA, this should be the base pretrained model. "
            "When using a merged / fully fine-tuned model, point directly to that directory."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help=(
            "Path to a LoRA checkpoint directory (e.g. ./checkpoints/checkpoint-500). "
            "Leave empty when --model_path already points to a merged model."
        ),
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path or URL to an input image.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path or URL to an input video.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt / question.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate (default: 512).",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype for model weights (default: bfloat16).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map for model loading, e.g. 'auto', 'cuda', 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for Qwen3-VL models (default: disabled).",
    )
    args = parser.parse_args()

    # Resolve torch dtype
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    # Determine which path holds the processor (prefer checkpoint, fall back to base)
    processor_path = args.model_path
    if args.checkpoint_path and os.path.exists(
        os.path.join(args.checkpoint_path, "preprocessor_config.json")
    ):
        processor_path = args.checkpoint_path

    print(f"Loading processor from {processor_path} ...")
    processor = AutoProcessor.from_pretrained(processor_path)

    print(f"Loading model from {args.model_path} ...")
    model, model_type = load_model(
        args.model_path, args.checkpoint_path, args.device, torch_dtype
    )
    print(f"Model type detected: {model_type}")

    messages = build_messages(args.prompt, args.image, args.video)

    print("Running inference ...")
    responses = run_inference(
        model,
        processor,
        model_type,
        messages,
        args.max_new_tokens,
        args.enable_thinking,
    )

    print("\n" + "=" * 60)
    print("Response:")
    print("=" * 60)
    for resp in responses:
        print(resp)


if __name__ == "__main__":
    main()
