import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

INIT_DIR = "./models/distilgpt2-2to4-init"
OUT_DIR  = "./models/distilgpt2-2to4-sft"
MASK_PATH = os.path.join(INIT_DIR, "masks.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ApplyMaskCallback(TrainerCallback):
    """
    每次 optimizer step 后重新把 mask 应用到 Linear 权重上，保证 2:4 稀疏结构不被破坏。
    """
    def __init__(self, masks):
        self.masks = masks  # dict: module_name -> bool mask (cpu)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and name in self.masks:
                    mask = self.masks[name].to(module.weight.device)
                    module.weight.data.mul_(mask)
        return control

def tokenize_dataset(tokenizer, block_size=256):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def _tokenize(examples):
        return tokenizer(examples["text"])

    tokenized = ds.map(_tokenize, batched=True, remove_columns=["text"])

    # 把 token 串拼起来切 block（标准 causal LM 处理方式）
    def _group_texts(examples):
        concatenated = {}
        for k in examples.keys():
            concatenated[k] = sum(examples[k], [])
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {}
        for k, t in concatenated.items():
            t = t[:total_length]
            result[k] = [t[i : i + block_size] for i in range(0, total_length, block_size)]
        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(_group_texts, batched=True)
    return lm_ds["train"], lm_ds["validation"]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(INIT_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(INIT_DIR).to(DEVICE)

    masks = torch.load(MASK_PATH, map_location="cpu")

    train_ds, val_ds = tokenize_dataset(tokenizer, block_size=256)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 训练建议：小学习率 + fp16 + 适度 steps
    args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=2000,
    logging_steps=50,

    do_eval=True,        # ✅ 开启评估
    eval_steps=200,      # ✅ 每200步 eval
    save_steps=200,      # ✅ 每200步保存

    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False,
)


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[ApplyMaskCallback(masks)],
    )

    trainer.train()

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Saved SFT model to:", OUT_DIR)

if __name__ == "__main__":
    main()