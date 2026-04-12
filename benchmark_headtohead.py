"""Head-to-head throughput benchmark: BitNet vs competing models on same hardware."""
import subprocess
import re
import json
import time
import os

LLAMA_CLI = "/home/cpinchington/bitnet.cpp/build/bin/llama-cli"

PROMPTS = [
    "The theory of general relativity explains that gravity is",
    "In machine learning, the backpropagation algorithm works by",
    "The process of photosynthesis in plants begins when sunlight",
    "Quantum computing differs from classical computing because qubits can",
    "The French Revolution of 1789 was triggered by a combination of",
    "To implement a binary search tree in Python, you first need to",
    "The human immune system fights infections through a process called",
    "Climate change is primarily driven by greenhouse gas emissions from",
]

def benchmark_model(model_path, model_name, n_tokens=256, threads=16, n_runs=8):
    """Run benchmark and return stats."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Model: {model_path}")
    print(f"Tokens: {n_tokens}, Threads: {threads}, Runs: {n_runs}")
    print(f"{'='*60}")

    results = []
    for i, prompt in enumerate(PROMPTS[:n_runs]):
        try:
            t0 = time.time()
            proc = subprocess.run(
                [LLAMA_CLI, "-m", model_path, "-p", prompt, "-n", str(n_tokens),
                 "-t", str(threads), "--temp", "0", "--no-display-prompt",
                 "--repeat-penalty", "1.1"],
                capture_output=True, text=True, timeout=120
            )
            t1 = time.time()

            stderr = proc.stderr + proc.stdout
            eval_match = re.search(
                r'eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(.*?([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second',
                stderr
            )
            prefill_match = re.search(
                r'prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(.*?([\d.]+)\s*tokens per second',
                stderr
            )

            if eval_match:
                gen_tok_s = float(eval_match.group(4))
                gen_ms_per_tok = float(eval_match.group(3))
                prefill_tok_s = float(prefill_match.group(3)) if prefill_match else 0

                results.append({
                    "gen_tok_s": gen_tok_s,
                    "gen_ms_per_tok": gen_ms_per_tok,
                    "prefill_tok_s": prefill_tok_s,
                    "wall_time": t1 - t0,
                })
                print(f"  Run {i+1}: {gen_tok_s:.1f} tok/s ({gen_ms_per_tok:.2f} ms/tok), prefill: {prefill_tok_s:.0f} tok/s")
            else:
                print(f"  Run {i+1}: FAILED to parse output")
                # Check for errors
                if "error" in stderr.lower():
                    error_lines = [l for l in stderr.split('\n') if 'error' in l.lower()]
                    print(f"    Error: {error_lines[0][:100] if error_lines else 'unknown'}")
        except subprocess.TimeoutExpired:
            print(f"  Run {i+1}: TIMEOUT")
        except Exception as e:
            print(f"  Run {i+1}: ERROR {e}")

    if results:
        avg_gen = sum(r["gen_tok_s"] for r in results) / len(results)
        avg_prefill = sum(r["prefill_tok_s"] for r in results) / len(results)
        avg_ms = sum(r["gen_ms_per_tok"] for r in results) / len(results)
        model_size_mb = os.path.getsize(model_path) / 1e6

        summary = {
            "name": model_name,
            "model_path": model_path,
            "model_size_mb": model_size_mb,
            "n_runs": len(results),
            "avg_gen_tok_s": avg_gen,
            "avg_prefill_tok_s": avg_prefill,
            "avg_ms_per_tok": avg_ms,
            "runs": results,
        }

        print(f"\n  SUMMARY: {avg_gen:.1f} tok/s gen, {avg_prefill:.0f} tok/s prefill, {avg_ms:.2f} ms/tok, {model_size_mb:.0f} MB")
        return summary

    return None


def main():
    models = []

    # BitNet b1.58 2B-4T (I2_S)
    bitnet_path = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    if os.path.exists(bitnet_path):
        models.append((bitnet_path, "BitNet b1.58 2B-4T (I2_S)"))

    # Competing models
    competing_dir = "/home/cpinchington/MedusaBitNet/models/competing"
    for name, filename, label in [
        ("Qwen2.5-1.5B", "qwen2.5-1.5b-instruct-q4_k_m.gguf", "Qwen2.5 1.5B (Q4_K_M)"),
        ("Llama-3.2-1B", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Llama 3.2 1B (Q4_K_M)"),
        ("Gemma-2-2B", "gemma-2-2b-it-Q4_K_M.gguf", "Gemma 2 2B (Q4_K_M)"),
    ]:
        path = os.path.join(competing_dir, filename)
        if os.path.exists(path):
            models.append((path, label))
        else:
            print(f"Skipping {label} — not downloaded yet")

    print(f"\n{'#'*60}")
    print(f"HEAD-TO-HEAD BENCHMARK")
    print(f"Hardware: AMD Ryzen AI MAX+ 395 (Strix Halo)")
    print(f"Threads: 16, Tokens: 256")
    print(f"Models: {len(models)}")
    print(f"{'#'*60}")

    all_results = []
    for path, label in models:
        result = benchmark_model(path, label)
        if result:
            all_results.append(result)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"HEAD-TO-HEAD RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<35} {'Gen tok/s':>10} {'Prefill':>10} {'ms/tok':>8} {'Size MB':>8}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: x["avg_gen_tok_s"], reverse=True):
        print(f"{r['name']:<35} {r['avg_gen_tok_s']:>10.1f} {r['avg_prefill_tok_s']:>10.0f} {r['avg_ms_per_tok']:>8.2f} {r['model_size_mb']:>8.0f}")

    # Save results
    with open("benchmark_headtohead.json", "w") as f:
        json.dump({"results": all_results, "hardware": "AMD Ryzen AI MAX+ 395 (Strix Halo)", "threads": 16}, f, indent=2)
    print(f"\nSaved to benchmark_headtohead.json")


if __name__ == "__main__":
    main()
