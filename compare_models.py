import os, time, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "dense": "./models/distilgpt2",
    "pruned30": "./models/distilgpt2-pruned30",
    "pruned2to4": "./models/distilgpt2-2to4-sft",
}

PROMPTS = [
    "Once upon a time, in a small village, there lived",
    "The future of artificial intelligence will depend on",
    "In 2025, the most important breakthrough in medicine was",
    "Write a short story about a robot that learns emotions:",
    "Explain in simple terms why the sky is blue:",
]

def distinct_n(texts, n=1):
    # distinct-1/2：不同 n-gram 占比
    total_ngrams = 0
    unique_ngrams = set()
    for t in texts:
        tokens = t.split()
        ngrams = list(zip(*[tokens[i:] for i in range(n)]))
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams

def repetition_rate(text):
    # 简单重复率：重复 token 占比（越高越容易重复）
    tokens = text.split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(tokens)

@torch.no_grad()
def compute_ppl(model, tokenizer, max_samples=200, max_length=256):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
    ds = ds.select(range(min(max_samples, len(ds))))

    losses = []
    for item in ds:
        enc = tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = model(**enc, labels=enc["input_ids"])
        losses.append(out.loss.item())

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    return ppl

@torch.no_grad()
def benchmark_generation(model, tokenizer, prompt="Hello", gen_len=128, runs=50, warmup=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # warmup
    for _ in range(warmup):
        _ = model.generate(**inputs, max_new_tokens=gen_len, do_sample=False)

    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.time()
    for _ in range(runs):
        _ = model.generate(**inputs, max_new_tokens=gen_len, do_sample=False)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    end = time.time()

    avg_time = (end - start) / runs
    tokens_per_sec = gen_len / avg_time
    return avg_time, tokens_per_sec

@torch.no_grad()
def generate_samples(model, tokenizer, prompts, gen_len=128):
    outputs = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt").to(DEVICE)
        out_ids = model.generate(
            **inp,
            max_new_tokens=gen_len,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.05,
        )
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        outputs.append(text)
    return outputs

def max_vram_mb():
    if DEVICE != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / 1024**2

def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    # distilgpt2 默认没有 pad_token，生成时建议设定
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(path)
    model.to(DEVICE)
    model.eval()
    return model, tok

def evaluate_model(name, path):
    print(f"\n===== Evaluating {name}: {path} =====")
    torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None

    model, tok = load_model(path)

    # PPL
    ppl = compute_ppl(model, tok)

    # speed
    prompt = "Hello world. " * 32
    latency, tps = benchmark_generation(model, tok, prompt=prompt, gen_len=128)

    # samples
    texts = generate_samples(model, tok, PROMPTS, gen_len=128)

    # text stats
    rep = sum(repetition_rate(t) for t in texts) / len(texts)
    d1 = distinct_n(texts, 1)
    d2 = distinct_n(texts, 2)

    vram = max_vram_mb()

    return {
        "name": name,
        "path": path,
        "ppl": ppl,
        "latency_s": latency,
        "tokens_per_s": tps,
        "max_vram_mb": vram,
        "rep_rate": rep,
        "distinct1": d1,
        "distinct2": d2,
        "samples": texts,
    }

def print_report(results):
    print("\n\n==================== FINAL REPORT ====================")
    for r in results:
        print(f"\n--- {r['name']} ---")
        print(f"Path: {r['path']}")
        print(f"PPL (WikiText-2): {r['ppl']:.3f}")
        print(f"Latency (128 new tokens): {r['latency_s']:.4f} s")
        print(f"Throughput: {r['tokens_per_s']:.2f} tokens/s")
        if r["max_vram_mb"] is not None:
            print(f"Max VRAM: {r['max_vram_mb']:.1f} MB")
        print(f"Repetition rate: {r['rep_rate']:.3f}")
        print(f"Distinct-1: {r['distinct1']:.3f}")
        print(f"Distinct-2: {r['distinct2']:.3f}")

    print("\n\n==================== SAMPLE OUTPUTS ====================")
    for i, p in enumerate(PROMPTS):
        print(f"\n\nPROMPT #{i+1}: {p}")
        for r in results:
            print(f"\n[{r['name']}]")
            print(r["samples"][i])

def main():
    all_results = []
    for name, path in MODEL_PATHS.items():
        all_results.append(evaluate_model(name, path))

    print_report(all_results)

if __name__ == "__main__":
    main()
