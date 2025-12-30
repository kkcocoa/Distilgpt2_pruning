import os
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./models/distilgpt2"
SPARSITY = 0.30  # 30% 稀疏

SAVE_DIR = f"./models/distilgpt2-pruned{int(SPARSITY*100)}"

def compute_sparsity(model):
    """统计模型中 Linear 权重的稀疏率"""
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    print(f"Applying unstructured L1 pruning with sparsity={SPARSITY} ...")

    # 对所有 Linear 层进行 L1 非结构化剪枝
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=SPARSITY)

    # 将 mask 合并到权重里（永久化）
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, "weight")

    sparsity_ratio = compute_sparsity(model)
    print(f"✅ Actual Linear weight sparsity: {sparsity_ratio*100:.2f}%")

    print("Saving pruned model...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Saved to:", SAVE_DIR)

if __name__ == "__main__":
    main()
