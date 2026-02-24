#!/usr/bin/env python3
"""Local GPU training script for DEXI drone LLM.

Fine-tunes a small instruct model using QLoRA + Unsloth on any NVIDIA GPU.
Equivalent to dexi_finetune.ipynb but runs standalone (no Colab).

Supported models (see config/models.json):
    - qwen2.5-1.5b  (default)
    - llama-3.2-1b
    - llama-3.2-3b

Requirements:
    pip install unsloth

Usage:
    python training/train_local.py
    python training/train_local.py --model llama-3.2-1b
    python training/train_local.py --model llama-3.2-3b --epochs 5 --batch-size 4
    python training/train_local.py --skip-test        # skip post-training test
    python training/train_local.py --skip-export       # skip GGUF export
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
CONFIG_DIR = TRAINING_DIR.parent / "dexi_llm" / "config"
DATASET_DIR = TRAINING_DIR / "dataset"


def main():
    with open(CONFIG_DIR / "models.json") as f:
        all_models = json.load(f)
    available_models = list(all_models.keys())

    parser = argparse.ArgumentParser(description="Train DEXI drone LLM locally")
    parser.add_argument("--model", type=str, default="qwen2.5-1.5b",
                        choices=available_models,
                        help=f"Model to train (default: qwen2.5-1.5b)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--skip-test", action="store_true", help="Skip post-training test")
    parser.add_argument("--skip-export", action="store_true", help="Skip GGUF export")
    args = parser.parse_args()

    # ---- Load config ----
    MODEL_NAME = args.model
    MODEL_CONFIG = all_models[MODEL_NAME]
    print(f"Model: {MODEL_NAME} ({MODEL_CONFIG['base_model']})")

    tpl = MODEL_CONFIG["chat_template"]

    with open(CONFIG_DIR / "tools.json") as f:
        TOOLS = json.load(f)
    with open(CONFIG_DIR / "system_prompt.txt") as f:
        SYSTEM_PROMPT = f.read().strip()

    tools_json = "\n".join(json.dumps(t, separators=(",", ":")) for t in TOOLS)
    SYSTEM_BLOCK = (
        f"{SYSTEM_PROMPT}\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tools_json}\n</tools>\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )
    print(f"  System block: {len(SYSTEM_BLOCK)} chars")

    # ---- Load model ----
    print(f"\nLoading base model: {MODEL_CONFIG['base_model']}...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["base_model"],
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.0f}M")

    # ---- Add LoRA adapters ----
    print("\nAdding LoRA adapters...")
    lora_cfg = MODEL_CONFIG["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.0f}M ({100 * trainable / total:.1f}%)")

    # ---- Load dataset ----
    print("\nLoading dataset...")
    import torch
    from datasets import load_dataset

    dataset = load_dataset("json", data_files={
        "train": str(DATASET_DIR / "train.jsonl"),
        "val": str(DATASET_DIR / "val.jsonl"),
    })

    def format_example(example):
        text = tpl["bos"]
        text += tpl["system_prefix"] + SYSTEM_BLOCK + tpl["system_suffix"]
        for msg in example["messages"]:
            role = msg["role"]
            if role == "user":
                text += tpl["user_prefix"] + msg["content"] + tpl["user_suffix"]
            elif role == "assistant":
                text += tpl["assistant_prefix"] + msg["content"] + tpl["assistant_suffix"]
        return {"text": text}

    dataset = dataset.map(format_example)
    print(f"  Train: {len(dataset['train'])}, Val: {len(dataset['val'])}")

    def formatting_func(example):
        return example["text"]

    # ---- Completion-only collator ----
    response_template_ids = tokenizer.encode(
        tpl["assistant_prefix"], add_special_tokens=False
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    class CompletionOnlyCollator:
        def __call__(self, features):
            input_ids_list = [f["input_ids"] for f in features]
            max_len = max(len(ids) for ids in input_ids_list)
            tpl_len = len(response_template_ids)

            batch_ids, batch_mask, batch_labels = [], [], []
            for ids in input_ids_list:
                pad_len = max_len - len(ids)
                labels = list(ids) + [-100] * pad_len

                for i in range(len(ids) - tpl_len + 1):
                    if ids[i : i + tpl_len] == response_template_ids:
                        labels[: i + tpl_len] = [-100] * (i + tpl_len)

                batch_ids.append(ids + [pad_id] * pad_len)
                batch_mask.append([1] * len(ids) + [0] * pad_len)
                batch_labels.append(labels)

            return {
                "input_ids": torch.tensor(batch_ids),
                "attention_mask": torch.tensor(batch_mask),
                "labels": torch.tensor(batch_labels),
            }

    # ---- Train ----
    print(f"\nTraining for {args.epochs} epochs...")
    from trl import SFTTrainer, SFTConfig

    os.environ["WANDB_DISABLED"] = "true"

    use_bf16 = model.get_input_embeddings().weight.dtype == torch.bfloat16

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        formatting_func=formatting_func,
        data_collator=CompletionOnlyCollator(),
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=10,
            eval_strategy="epoch",
            output_dir=MODEL_CONFIG["training"]["lora_dir"],
            seed=42,
            report_to="none",
            max_seq_length=args.max_seq_length,
        ),
    )

    print(f"  Precision: {'bf16' if use_bf16 else 'fp16'}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    results = trainer.train()
    print(f"\nTraining done! Loss: {results.training_loss:.4f}")

    # ---- Test ----
    if not args.skip_test:
        print(f"\n{'='*70}")
        print("  POST-TRAINING TEST")
        print(f"{'='*70}")

        FastLanguageModel.for_inference(model)

        test_cases = [
            ("make the led green", True),
            ("fly forward 2 meters", True),
            ("what's my battery level?", True),
            ("do a flip", True),
            ("set servo 3 to 90 degrees", True),
            ("what topics are available?", True),
            ("start recording video", True),
            ("play some music", False),
        ]

        def build_prompt(user_msg):
            messages = [
                {"role": "system", "content": SYSTEM_BLOCK},
                {"role": "user", "content": user_msg},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        passed = 0
        for prompt, expect_tool in test_cases:
            input_text = build_prompt(prompt)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
            )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            has_tool = "<tool_call>" in response
            correct = has_tool == expect_tool
            if correct:
                passed += 1
            status = "PASS" if correct else "FAIL"
            expect_str = "tool_call" if expect_tool else "refusal"
            got_str = "tool_call" if has_tool else "text"

            print(f"\n[{status}] \"{prompt}\"")
            print(f"  Expected: {expect_str} | Got: {got_str}")
            print(f"  Response: {response[:200]}")

        print(f"\n{'='*70}")
        print(f"  Score: {passed}/{len(test_cases)} passed")
        print(f"{'='*70}")

    # ---- Export GGUF ----
    if not args.skip_export:
        print("\nExporting to GGUF...")
        merged_dir = MODEL_CONFIG["training"]["merged_dir"]
        gguf_dir = MODEL_CONFIG["training"]["gguf_dir"]

        print(f"  Merging LoRA weights to {merged_dir}...")
        model.save_pretrained_merged(merged_dir, tokenizer)

        print(f"  Exporting to GGUF (Q4_K_M) in {gguf_dir}...")
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method="q4_k_m",
        )

        search_dirs = [gguf_dir, f"{gguf_dir}_gguf"]
        gguf_files = []
        for d in search_dirs:
            found = glob.glob(f"{d}/*.gguf")
            if found:
                print(f"\n  Found .gguf files in {d}/:")
                for fp in found:
                    size_mb = os.path.getsize(fp) / (1024 * 1024)
                    print(f"    {os.path.basename(fp)} ({size_mb:.0f} MB)")
                gguf_files.extend(found)

        gguf_files = sorted(gguf_files, key=os.path.getsize, reverse=True)
        if gguf_files:
            gguf_path = gguf_files[0]
            size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
            gguf_name = MODEL_CONFIG["gguf_name"]
            print(f"\nGGUF ready: {gguf_path} ({size_mb:.0f} MB)")
            print(f"\nDeploy:")
            print(f"  cp {gguf_path} dexi_ws/src/dexi_llm/models/{gguf_name}")
            print(f"  # Or SCP to Pi:")
            print(f"  scp {gguf_path} dexi@192.168.68.59:~/dexi_ws/src/dexi_llm/models/{gguf_name}")
        else:
            print(f"\nERROR: No .gguf files found in {search_dirs}")
            sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
