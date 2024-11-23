"""
envd build -f :serving --output type=image,name=ghcr.io/unionai-oss/modelz-flyte-llama-serving:v0,push=true
"""

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import huggingface_hub as hh
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer


@dataclass
class ServingConfig:
    model_path: str
    adapter_path: str
    model_max_length: int = 1024
    max_new_tokens: int = 2048
    padding: str = "right"
    device_map: str = "auto"
    use_4bit: bool = False
    hf_token: Optional[str] = None


def load_pipeline(
        config: ServingConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load merged model and tokenizer directly."""
    if config.hf_token:
        print(f"Using HF token: {config.hf_token[:8]}...")
        hh.login(token=config.hf_token)

    # Load tokenizer from merged model path
    print(f"Loading tokenizer from model path: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        token=config.hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    # 保持与 train 一致的加载参数
    load_model_params = {
        "use_auth_token": config.hf_token,
        "torch_dtype": dtype,
        "device_map": config.device_map,
    }
    if config.use_4bit:
        load_model_params = {
            **load_model_params,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,
            ),
        }

    print(f"Loading merged model from: {config.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **load_model_params,
    ).to("cuda")

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")

    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", default=False)
    args = parser.parse_args()

    config = ServingConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        adapter_path="unionai/FlyteLlama-v0-7b-hf-flyte-repos",
        device_map=None,
    )
    print("Loading model and tokenizer...")
    model, tokenizer = load_pipeline(config)

    print("Generating...")
    prompts = ["The code below shows a basic Flyte workflow"]
    print(prompts[0], end="", flush=True)

    prev_msg = prompts[0]
    if args.stream:
        tokens = tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
        )
        inputs = tokens["input_ids"]
        for i in range(100):
            inputs = model.generate(
                inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=config.max_new_tokens,
            )

            if inputs.shape[-1] >= config.model_max_length:
                inputs = inputs[:, -config.model_max_length:]

            msg = tokenizer.decode(inputs[0])
            print_msg = msg[len(prev_msg):]
            print(print_msg, end="", flush=True)
            prev_msg = msg
    else:
        inputs = tokenizer(prompts, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=40,
            pad_token_id=tokenizer.eos_token_id,
        )
        for output in outputs:
            print(tokenizer.decode(output, skip_special_tokens=True))
