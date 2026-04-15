"""End-to-end C++ benchmark: llama-medusa vs vanilla llama-cli on the same
prompts and same GGUF.

This is THE benchmark that matters — it's what anyone reproducing the
MedusaBitNet work will run on their own hardware. Measures real wall-clock
tokens-per-second and real accepted speculations reported by llama-medusa.
No Python-side simulation, no backbone-free head eval.
"""
import argparse
import re
import subprocess
import time
from pathlib import Path


LLAMA_CLI = "/home/cpinchington/bitnet.cpp/build/bin/llama-cli"
LLAMA_MED = "/home/cpinchington/bitnet.cpp/build/bin/llama-medusa"

DEFAULT_PROMPTS = [
    "The capital of France is",
    "A short explanation of photosynthesis:",
    "The three laws of motion state that",
    "Here is a simple Python function to compute the nth Fibonacci number:",
    "The French Revolution began in",
    "Machine learning is a field of computer science that",
    "The Pythagorean theorem states that in a right triangle,",
    "Q: What is the speed of light?\nA:",
]


def run_vanilla(gguf: str, prompt: str, n: int, threads: int) -> dict:
    t0 = time.time()
    r = subprocess.run(
        [LLAMA_CLI, "-m", gguf, "-p", prompt, "-n", str(n),
         "--temp", "0", "-t", str(threads)],
        capture_output=True, text=True, timeout=300,
    )
    wall = time.time() - t0
    # Parse llama.cpp eval tok/s
    m = re.search(r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)", r.stderr)
    eval_toks = int(m.group(1)) if m else 0
    eval_rate = float(m.group(2)) if m else 0.0
    return {
        "wall_s": wall,
        "eval_toks": eval_toks,
        "eval_tok_per_s": eval_rate,
        "ok": r.returncode == 0,
    }


def run_medusa(gguf: str, prompt: str, n: int, threads: int) -> dict:
    t0 = time.time()
    r = subprocess.run(
        [LLAMA_MED, "-m", gguf, "-p", prompt, "-n", str(n), "-t", str(threads)],
        capture_output=True, text=True, timeout=300,
    )
    wall = time.time() - t0
    # Parse "[medusa] steps=X  accepted_speculations=Y  mean_accept_per_step=Z/K  generated=N"
    m = re.search(r"\[medusa\]\s*steps=(\d+)\s*accepted_speculations=(\d+)\s*mean_accept_per_step=([\d.]+)/(\d+)\s*generated=(\d+)", r.stderr + r.stdout)
    if not m:
        return {"wall_s": wall, "ok": False, "err": (r.stderr+r.stdout)[-300:]}
    steps, accepted, mean_acc, k, generated = m.groups()
    return {
        "wall_s": wall,
        "steps": int(steps),
        "accepted": int(accepted),
        "mean_accept_per_step": float(mean_acc),
        "n_heads": int(k),
        "generated": int(generated),
        "tok_per_s_wall": int(generated) / wall,
        "ok": r.returncode == 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla", required=True, help="GGUF path for vanilla llama-cli")
    p.add_argument("--medusa",  required=True, help="GGUF path for llama-medusa (with heads)")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--prompts", nargs="*", default=None)
    args = p.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    print(f"Benchmarking on {len(prompts)} prompts, n={args.n}, threads={args.threads}")
    print(f"  vanilla: {args.vanilla}")
    print(f"  medusa:  {args.medusa}\n")

    results_v = []
    results_m = []
    for i, prm in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prm[:60]!r}")
        v = run_vanilla(args.vanilla, prm, args.n, args.threads)
        m = run_medusa(args.medusa, prm, args.n, args.threads)
        print(f"  vanilla: {v.get('eval_tok_per_s', 0):.2f} tok/s ({v.get('eval_toks', 0)} tokens, wall {v['wall_s']:.2f}s)")
        if m.get("ok"):
            print(f"  medusa:  {m['tok_per_s_wall']:.2f} tok/s wall, steps={m['steps']} accepted={m['accepted']} "
                  f"mean={m['mean_accept_per_step']:.2f}/{m['n_heads']}  ({m['generated']} tokens, wall {m['wall_s']:.2f}s)")
        else:
            print(f"  medusa ERROR: {m.get('err', '??')[:200]}")
        results_v.append(v); results_m.append(m)

    # Aggregate
    print("\n=== SUMMARY ===")
    v_avg = sum(r.get("eval_tok_per_s", 0) for r in results_v) / max(1, len(results_v))
    ok_m = [r for r in results_m if r.get("ok")]
    if ok_m:
        m_tps_avg = sum(r["tok_per_s_wall"] for r in ok_m) / len(ok_m)
        m_acc_avg = sum(r["accepted"] for r in ok_m) / sum(r["steps"] for r in ok_m) if sum(r["steps"] for r in ok_m) else 0
        total_steps = sum(r["steps"] for r in ok_m)
        total_accepted = sum(r["accepted"] for r in ok_m)
        total_generated = sum(r["generated"] for r in ok_m)
        print(f"vanilla mean tok/s:        {v_avg:.2f}")
        print(f"medusa wall tok/s:         {m_tps_avg:.2f}  (speedup {m_tps_avg/v_avg:.2f}x)")
        print(f"medusa acceptance rate:    {total_accepted}/{total_steps} "
              f"= {100*total_accepted/total_steps:.1f}% of verify steps")
        print(f"medusa tokens per step:    {total_generated/total_steps:.2f}  "
              f"(1.0 = no spec, up to {ok_m[0]['n_heads']+1} max with {ok_m[0]['n_heads']} heads)")


if __name__ == "__main__":
    main()
