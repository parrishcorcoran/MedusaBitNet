"""Compare Python head predictions to llama-cli greedy generation on the
SAME GGUF, for the same prompt. If they agree but llama-medusa still shows
0 acceptance, the bug is in llama-medusa's C++ head forward pass, not the
head weights or the training.
"""
import subprocess, numpy as np, torch
from transformers import AutoTokenizer
from model import MedusaHeads

HIDDEN = 2560
GGUF = "/home/cpinchington/models/official-bitnet/ggml-model-i2_s.gguf"
HIDDEN_DUMP = "/home/cpinchington/bitnet.cpp/build/bin/llama-hidden-dump"
LLAMA_CLI = "/home/cpinchington/bitnet.cpp/build/bin/llama-cli"
CKPT = "checkpoints/pilot_result_norm/medusa_heads_step500.pt"
TOK_DIR = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T"
PROMPT = "The capital of France is"


def main():
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    # Tokenize to IDs with BOS, same as llama-cli.
    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=True).input_ids[0].tolist()
    if ids[0] != 128000:
        ids = [128000] + ids
    print(f"prompt tokens ({len(ids)}): {ids}")
    np.array(ids, dtype=np.uint32).tofile("/tmp/cmp_tokens.bin")

    # 1. Dump result_norm from GGUF for the prompt
    subprocess.run([HIDDEN_DUMP, "-m", GGUF, "-p", "x",
                    "--tokens-file", "/tmp/cmp_tokens.bin",
                    "--tensor", "result_norm",
                    "--dump-out", "/tmp/cmp_norm.bin", "-t", "16"],
                   check=True, capture_output=True)
    h = np.fromfile("/tmp/cmp_norm.bin", dtype=np.float32).reshape(-1, HIDDEN)
    print(f"hidden states dumped: {h.shape}")

    # 2. Get greedy next-tokens via llama-cli with n=10
    r = subprocess.run([LLAMA_CLI, "-m", GGUF, "-p", PROMPT, "-n", "10",
                        "--temp", "0", "-t", "16"],
                       capture_output=True, text=True, timeout=120)
    full = r.stdout
    # llama-cli echoes the prompt then generated text on its own line(s).
    gen = full.split(PROMPT, 1)[-1].strip().split("\n\n")[0]
    print(f"greedy gen: {gen!r}")
    gen_ids = tok(gen, add_special_tokens=False).input_ids
    print(f"greedy gen ids (first 5): {gen_ids[:5]}")

    # 3. Run head on last prompt position's hidden state, get top-1
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, 128256, 1, 1, dtype=torch.bfloat16)
    heads.load_state_dict(ckpt["heads"])
    heads.eval()
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    with torch.no_grad():
        h_t = torch.from_numpy(h).to(torch.bfloat16).unsqueeze(0)  # [1, T, H]
        logits = heads(h_t, lm_head)  # [1, T, 1, V]
        last_logits = logits[0, -1, 0]  # [V] -- at last prompt position
        topk = torch.topk(last_logits.float(), 5)

    print(f"\nhead-0 top-5 at last prompt pos:")
    for i, (v, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
        print(f"  {i}. {tok.decode([idx])!r:20s}  id={idx}  logit={v:.3f}")

    # The greedy first token is what base would pick. Head-0 should match.
    if gen_ids:
        match = (gen_ids[0] == topk.indices[0].item())
        print(f"\nbase greedy first token: {tok.decode([gen_ids[0]])!r} (id={gen_ids[0]})")
        print(f"head-0 top-1:           {tok.decode([topk.indices[0].item()])!r}")
        print(f"MATCH: {match}")


if __name__ == "__main__":
    main()
