import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./models/distilgpt2"
SAVE_DIR = "./models/distilgpt2-2to4-init"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_2to4_mask(weight: torch.Tensor):
    """
    weight: [out, in]
    mask:  same shape, True=keep, False=zero
    2:4 along the last dimension (in_features) in groups of 4.
    """
    out_dim, in_dim = weight.shape
    in_trim = (in_dim // 4) * 4

    w_main = weight[:, :in_trim]
    w_rest = weight[:, in_trim:] if in_trim < in_dim else None

    w = w_main.view(out_dim, -1, 4)
    abs_w = w.abs()

    topk_vals, topk_idx = torch.topk(abs_w, k=2, dim=-1, largest=True)

    mask_main = torch.zeros_like(w, dtype=torch.bool)
    mask_main.scatter_(-1, topk_idx, True)
    mask_main = mask_main.view(out_dim, in_trim)

    if w_rest is not None:
        mask_rest = torch.ones_like(w_rest, dtype=torch.bool)  # 不剪剩余尾巴
        mask = torch.cat([mask_main, mask_rest], dim=1)
    else:
        mask = mask_main

    return mask

def compute_sparsity_linear(model):
    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total if total else 0.0

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    masks = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight.data
                mask = build_2to4_mask(w)
                module.weight.data = w * mask.to(w.device)
                masks[name] = mask.cpu()

    print(f"2:4 init sparsity (Linear): {compute_sparsity_linear(model)*100:.2f}%")

    model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    torch.save(masks, os.path.join(SAVE_DIR, "masks.pt"))
    print("Saved:", SAVE_DIR)

if __name__ == "__main__":
    main()
