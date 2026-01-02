import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

try:
    import mlflow
except Exception:  # noqa: BLE001
    mlflow = None

BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceH4/zephyr-7b-beta")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "artifacts/lora"))


def format_example(prompt: str, spec: Dict) -> str:
    prefix = (
        "Convert the following scheduling request into JSON with fields "
        "tasks, resources, must_assign_all_tasks. JSON:"
    )
    return f"{prefix}\nRequest: {prompt}\nJSON:"


def load_data() -> Dict[str, List[Dict]]:
    data_files = {"train": "data/train.jsonl", "validation": "data/val.jsonl"}
    for path in data_files.values():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. Run training/synth_data.py first to generate synthetic data."
            )
    return load_dataset("json", data_files=data_files)


def build_model():
    use_cuda = torch.cuda.is_available()
    quant_config = None
    if use_cuda:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except Exception:  # noqa: BLE001
            quant_config = None
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto" if use_cuda else None,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def build_collator(tokenizer):
    def collate(batch):
        input_ids_list = []
        attention_masks = []
        labels_list = []
        for row in batch:
            prompt_text = format_example(row["prompt"], row["json_spec"])
            target_json = json.dumps(row["json_spec"], separators=(",", ":"))
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            target_ids = tokenizer(target_json, add_special_tokens=False).input_ids + [
                tokenizer.eos_token_id
            ]
            input_ids = prompt_ids + target_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + target_ids
            input_ids_list.append(torch.tensor(input_ids))
            attention_masks.append(torch.tensor(attention_mask))
            labels_list.append(torch.tensor(labels))
        input_ids_list = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels_list = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_masks,
            "labels": labels_list,
        }

    return collate


def main():
    dataset = load_data()
    model, tokenizer = build_model()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        report_to=["none"],
    )

    data_collator = build_collator(tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    if mlflow and os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.start_run(run_name="optim-copilot-lora")
        mlflow.log_params({"base_model": BASE_MODEL, "output_dir": str(OUTPUT_DIR)})

    trainer.train()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")

    if mlflow and os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="lora_adapter")
        mlflow.end_run()


if __name__ == "__main__":
    main()
