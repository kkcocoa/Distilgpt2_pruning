import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from transformers import BitsAndBytesConfig

MODEL_PATH = "./models/distilgpt2-pruned30"

PROMPT = "The Counter Strike is a game,"
MAX_NEW_TOKENS = 256
RUNS = 10   # 测试次数（建议 10~30）

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

from transformers.pytorch_utils import Conv1D

def report_sparsity(model):
    total = 0
    zeros = 0

    for m in model.modules():
        # GPT2 的大部分权重在 Conv1D
        if isinstance(m, Conv1D):
            w = m.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
        # lm_head 是 Linear
        elif isinstance(m, torch.nn.Linear):
            w = m.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()

    if total == 0:
        print("No weights counted for sparsity.")
        return

    print(f"Sparsity (zero ratio): {zeros/total*100:.2f}%  (zeros={zeros}, total={total})")


GEN_KWARGS = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.15,
)

def timed_generate(model, inputs):
    """返回生成输出 + elapsed（严格同步后的计时）"""
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **GEN_KWARGS)
    torch.cuda.synchronize()
    t1 = time.time()
    return output_ids, (t1 - t0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH).to(device)
    model.eval()
    report_sparsity(model)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    # warmup
    print("Warming up...")
    _ = model.generate(**inputs, max_new_tokens=32)
    torch.cuda.synchronize()

    # reset peak memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    tps_list = []

    print(f"Benchmarking {RUNS} runs...")
    for i in range(RUNS):
        output_ids, elapsed = timed_generate(model, inputs)

        gen_tokens = output_ids.shape[-1] - inputs["input_ids"].shape[-1]
        tps = gen_tokens / elapsed

        times.append(elapsed)
        tps_list.append(tps)

        print(f"Run {i+1:02d}: time={elapsed:.3f}s, tokens={gen_tokens}, tps={tps:.2f}")

    # summary
    times = np.array(times)
    tps_list = np.array(tps_list)

    print("\n=== Summary ===")
    print(f"Mean time: {times.mean():.3f}s  |  P95 time: {np.percentile(times,95):.3f}s")
    print(f"Mean TPS : {tps_list.mean():.2f}  |  P95 TPS : {np.percentile(tps_list,5):.2f} (lower is worse)")

    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU memory allocated: {peak_mem:.1f} MiB")

    # show one generated sample
    sample_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== Sample Output ===")
    print(sample_text)

    print(next(model.parameters()).device)
    #print(model.hf_device_map)
    print(model)


if __name__ == "__main__":
    main()
