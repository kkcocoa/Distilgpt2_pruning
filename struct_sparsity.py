import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./models/distilgpt2"
SAVE_DIR = "./models/distilgpt2-pruned2to4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def apply_2to4_pruning(weight: torch.Tensor):
    """
    对 weight 做 2:4 structured pruning
    weight shape: [out_features, in_features]
    每连续4个元素保留|w|最大的2个，其余置0
    """
    # 确保 in_features 是 4 的倍数
    out_dim, in_dim = weight.shape
    if in_dim % 4 != 0:
        # 只处理可整除部分，剩下的保留不剪（你也可以选择 pad）
        in_dim_trim = (in_dim // 4) * 4
        w_main = weight[:, :in_dim_trim]
        w_rest = weight[:, in_dim_trim:]
    else:
        w_main = weight
        w_rest = None

    # reshape: [out, in/4, 4]
    w = w_main.view(out_dim, -1, 4)
    abs_w = w.abs()

    # 找每组绝对值最小的2个 -> mask=0
    # topk 保留最大2个，所以其余置0
    topk_vals, topk_idx = torch.topk(abs_w, k=2, dim=-1, largest=True)

    mask = torch.zeros_like(w, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)

    pruned = w * mask

    pruned = pruned.view(out_dim, -1)

    if w_rest is not None:
        pruned = torch.cat([pruned, w_rest], dim=1)

    return pruned

def compute_sparsity(model):
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
    model.to(DEVICE)

    print("Applying 2:4 structured pruning (50% sparsity in each group of 4)...")

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight.data
                module.weight.data = apply_2to4_pruning(w)

    sparsity_ratio = compute_sparsity(model)
    print(f"✅ Actual Linear weight sparsity: {sparsity_ratio*100:.2f}%")

    print("Saving pruned model...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Saved to:", SAVE_DIR)

if __name__ == "__main__":
    main()
