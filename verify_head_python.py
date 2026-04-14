"""Sanity check: run pilot head in Python on GGUF-dumped hidden states and
check if its top-1 prediction matches the greedy next-token from the same GGUF.

If Python predicts well but llama-medusa shows 0 acceptance, the bug is in
the C++ head integration. If Python also predicts poorly, the head or
merger is wrong.
"""
import numpy as np
import subprocess
import torch
from model import MedusaHeads

HIDDEN = 2560

def gguf_dump(tensor, gguf_path, prompt, out_path, tokens_file=None):
    args = ["/home/cpinchington/bitnet.cpp/build/bin/llama-hidden-dump",
            "-m", gguf_path, "-p", prompt,
            "--tensor", tensor, "--dump-out", out_path, "-t", "16"]
    if tokens_file:
        args += ["--tokens-file", tokens_file]
    subprocess.run(args, capture_output=True, check=True)


def main():
    prompt = "The capital of France is Paris. Paris is a city"
    gguf = "/home/cpinchington/models/official-bitnet/ggml-model-i2_s.gguf"

    # 1. Dump GGUF 'result_norm' hidden state (same as llama-medusa's h_final)
    gguf_dump("result_norm", gguf, prompt, "/tmp/vh_norm.bin")
    # 2. Dump GGUF logits via... well, we need next-token. Use llama-cli with greedy.
    #    Run llama-cli to see what next tokens are greedy-generated.
    r = subprocess.run(
        ["/home/cpinchington/bitnet.cpp/build/bin/llama-cli",
         "-m", gguf, "-p", prompt, "-n", "10", "--temp", "0", "-t", "16"],
        capture_output=True, text=True, timeout=120,
    )
    print("greedy output:", r.stdout.split(prompt)[-1][:200])

    # 3. Load pilot head + lm_head
    ckpt = torch.load("checkpoints/pilot_result_norm/medusa_heads_step500.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, 128256, 1, 1, dtype=torch.bfloat16).to("cpu")
    heads.load_state_dict(ckpt["heads"])
    heads.eval()
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)  # [V, H]

    # 4. Read hidden state, run head, decode prediction at final position
    norm = np.fromfile("/tmp/vh_norm.bin", dtype=np.float32)
    # Dump was [H, T] in ggml layout (ne[0]=H, ne[1]=T), byte-layout token-major
    # (tok0[h0..h-1], tok1[...]) after our fix. Let me verify shape.
    n_toks = norm.size // HIDDEN
    print(f"hidden dumped: {n_toks} tokens, shape [{n_toks},{HIDDEN}]")
    h = torch.from_numpy(norm.reshape(n_toks, HIDDEN)).to(torch.bfloat16).unsqueeze(0)

    with torch.no_grad():
        logits = heads(h, lm_head)  # [1, T, 1, V]
    preds = logits.argmax(-1).squeeze()  # [T] if k=1 else [T, k]
    print(f"head-0 predictions at last 5 positions: {preds[-5:].tolist()}")

    # Also get base greedy at each position by running llama-cli with tokens
    # Actually simpler: use transformers tokenizer to decode the pred IDs
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T")
    print("head-0 predicted tokens:", [tok.decode([t]) for t in preds[-5:].tolist()])


if __name__ == "__main__":
    main()
