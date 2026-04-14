"""5-minute diagnostic: does HF layer-30 hidden state match GGUF l_out-29?

If cosine similarity is near 1.0 for each token, then HF-trained Medusa heads
should work on GGUF hidden states — and the 0-acceptance bug is elsewhere.
If cosine is low, retraining on GGUF hidden states is justified.
"""
import numpy as np
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T"
GGUF = "/home/cpinchington/models/official-bitnet/ggml-model-i2_s.gguf"
HIDDEN_DUMP = "/home/cpinchington/bitnet.cpp/build/bin/llama-hidden-dump"
PROMPT = "The capital of France is Paris."


def gguf_hidden(tensor_name: str) -> np.ndarray:
    out = "/tmp/diag.bin"
    subprocess.run(
        [HIDDEN_DUMP, "-m", GGUF, "-p", PROMPT,
         "--tensor", tensor_name, "--dump-out", out, "-t", "16"],
        check=True, capture_output=True,
    )
    raw = np.fromfile(out, dtype=np.float32)
    # Shape is [2560, n_tokens]. Reshape to [n_tokens, 2560].
    return raw.reshape(-1, 2560)


def hf_hidden() -> tuple[np.ndarray, list[int]]:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # Use the HF hub ID (not local dir) — local config lacks the unpack glue
    # referenced via auto_map. The hub variant pulls the proper modeling files.
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/bitnet-b1.58-2B-4T",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    # Match how llama.cpp tokenizes: add_bos=True (default).
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    # hidden_states is tuple of n_layer+1 tensors of shape [1, T, H].
    # hidden_states[-1] is the final layer output (what Medusa heads are trained on).
    h = out.hidden_states[-1][0].to(torch.float32).cpu().numpy()  # [T, H]
    return h, ids[0].tolist()


def cos_per_token(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a, b: [T, H] — cosine per row.
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return num / den


def main():
    print("=== loading HF model for hidden states ===")
    hf_h, hf_tokens = hf_hidden()
    print(f"HF  hidden shape: {hf_h.shape}  tokens: {hf_tokens}")

    for tname in ["l_out-29", "norm"]:
        print(f"\n=== GGUF tensor: {tname} ===")
        try:
            g = gguf_hidden(tname)
        except subprocess.CalledProcessError as e:
            print(f"  dump failed: {e.stderr.decode()[:200]}")
            continue
        print(f"  GGUF shape: {g.shape}")
        cos = cos_per_token(hf_h, g)
        mse = ((hf_h[:len(cos)] - g[:len(cos)])**2).mean(1)
        norm_ratio = np.linalg.norm(g[:len(cos)], axis=1) / (np.linalg.norm(hf_h[:len(cos)], axis=1) + 1e-9)
        print(f"  per-token cosine: min={cos.min():.4f} mean={cos.mean():.4f} max={cos.max():.4f}")
        print(f"  per-token MSE:    min={mse.min():.4f} mean={mse.mean():.4f} max={mse.max():.4f}")
        print(f"  norm ratio (gguf/hf): min={norm_ratio.min():.3f} mean={norm_ratio.mean():.3f} max={norm_ratio.max():.3f}")


if __name__ == "__main__":
    main()
