"""Train Flyte Llama."""

import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses_json import dataclass_json, DataClassJsonMixin

import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from sky_llama.dataloader import get_dataset


transformers.logging.set_verbosity_debug()


@dataclass
class HuggingFaceModelCard(DataClassJsonMixin):
    language: List[str]
    license: str  # valid licenses can be found at https://hf.co/docs/hub/repositories-licenses
    tags: List[str]


@dataclass
class PublishConfig(DataClassJsonMixin):
    repo_id: str
    readme: Optional[str] = None
    model_card: Optional[HuggingFaceModelCard] = None


@dataclass
class TrainerConfig(DataClassJsonMixin):
    model_path: str = "codellama/CodeLlama-7b-hf"
    data_dir: str = "./sky_llama/dataset"
    output_dir: str = "./sky_llama/output"
    checkpoint_dir: Optional[str] = None
    num_epochs: int = 20
    max_steps: int = -1
    batch_size: int = 8
    test_size: float = 0.01
    model_max_length: int = 1024
    seed: int = 41
    report_to: str = "none"
    device_map: Optional[str] = None
    gradient_accumulation_steps: int = 8
    padding: str = "right"
    dataloader_num_proc: int = 1
    use_fp16: bool = False
    use_4bit: bool = False
    use_qlora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj"])
    lora_dropout: float = 0.05
    debug: bool = False
    publish_config: Optional[PublishConfig] = field(default=None)


def trainer_config() -> TrainerConfig:
    trainer_config = TrainerConfig(
        output_dir="./output",               # Directory to save the model and logs
        checkpoint_dir=None,                 # Directory to resume from a checkpoint
        num_epochs=3,                        # Increased to allow more training epochs
        max_steps=100,                       # Increased max steps for better convergence
        test_size=0.1,                       # Increased test size for more robust evaluation
        report_to="wandb",                   # Logging with Weights & Biases
        dataloader_num_proc=4,               # Reduced to 4 to avoid CPU contention
        use_fp16=True,                       # Enable mixed precision (16-bit floating-point)
        use_4bit=True,                       # Enable 4-bit precision
        use_qlora=True,                      # Enable QLoRA for fine-tuning
        padding="left",                      # Padding strategy (left padding for LLaMA)
        lora_target_modules=["q_proj", "v_proj"],  # Modules to target for LoRA fine-tuning
        batch_size=4,                        # Added batch size for per-device training
        gradient_accumulation_steps=16,      # Accumulate gradients for larger effective batch size
        # device_map={"": torch.cuda.current_device()},  # Load model on the current device
        device_map='auto',                   # Load model on the current device
    )
    return trainer_config

def train(
    hf_auth_token: Optional[str] = None,
    **kwargs,
):
    print("Training model...")
    # init_process_group(backend="nccl", init_method="env://")
    config = trainer_config()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side=config.padding,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        dtype = torch.float32
    
    init_process_group(backend="nccl", init_method="env://")

    # load pre-trained model
    load_model_params = {
        **kwargs,
        "use_auth_token": hf_auth_token,
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
                bnb_4bit_compute_dtype=dtype,
            ),
        }
        

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **load_model_params,
    )
    model = model.to(local_rank)

    optim = "adamw_torch"
    if config.use_qlora:
        optim = "paged_adamw_8bit"
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        print("LORA Config:")
        print(lora_config)
        model.print_trainable_parameters()

    def tokenize(examples):
        tokens = tokenizer(
            [f"{t}{tokenizer.eos_token}" for t in examples['text']],
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
        )
        return tokens

    limit = 5 if config.debug else None
    dataset = (
        get_dataset(
            Path(config.data_dir).expanduser(),
            num_proc=config.dataloader_num_proc,
            limit=limit,
            block_size=config.model_max_length,
            skip_by=config.model_max_length,
        )
        .map(tokenize, batched=True, num_proc=config.dataloader_num_proc)
    )

    # Remove unnecessary columns
    columns_to_remove = [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
    dataset = dataset.remove_columns(columns_to_remove)

    print(f"Dataset size: {len(dataset)}")
    dataset_splits = dataset.train_test_split(
        test_size=config.test_size, seed=config.seed
    )

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=3e-4,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=config.use_fp16,
        half_precision_backend="auto",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=0,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        logging_steps=1,
        optim=optim,
        report_to=config.report_to,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=config.checkpoint_dir)
    eval_results = trainer.evaluate(eval_dataset=dataset_splits["test"])
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.save_model(training_args.output_dir)