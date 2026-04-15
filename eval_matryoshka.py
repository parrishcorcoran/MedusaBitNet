"""Evaluate matryoshka pilot head on held-out sequences.

Compares matryoshka-trained head at each rank against:
  - SVD-truncated full-rank head at same rank
  - Full-rank head itself
"""
import numpy as np
import torch
import torch.nn.functional as F

SEQ_LEN = 2048; HIDDEN = 2560


def head_forward(hidden, W_in, W_out):
    x = hidden.float()
    return x + F.silu(x @ W_in.float()) @ W_out.float()


def main():
    # Load matryoshka head
    mat = torch.load("checkpoints/matryoshka_pilot/head_step500.pt",
                     map_location="cpu", weights_only=True)
    W_in_m = mat["W_in"]
    W_out_m = mat["W_out"]
    print(f"[mat] W_in {tuple(W_in_m.shape)}, W_out {tuple(W_out_m.shape)}")

    # Load original full-rank head-0 for comparison
    full = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    w_in_full = full["heads"]["w_in"][0, 0].float()   # [H, H]
    w_out_full = full["heads"]["w_out"][0, 0].float()

    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    # Held-out sequences (different from what matryoshka pilot saw)
    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    # pilot trained on seqs 0-4 from pilot bin. hidden_gguf_v2.bin is 1000 seqs;
    # use seqs 20-35 (not in pilot) as held-out
    test_seqs = list(range(20, 36))

    ranks = [32, 64, 128, 256, 512, 1024, 2560]

    all_hidden = []
    all_vpred = []
    for si in test_seqs:
        off = si * per_seq
        chunk = hidden_mm[off:off + per_seq]
        h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                   .view(SEQ_LEN, HIDDEN))
        vlogits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
        vpred = vlogits.float().argmax(dim=-1)
        all_hidden.append(h)
        all_vpred.append(vpred)

    print(f"\n=== Held-out evaluation ({len(test_seqs)} seqs) ===")
    print(f"{'rank':>6} {'matryoshka':>12} {'SVD-truncated':>14} {'improvement':>12}")
    print("-" * 50)

    def truncate_svd(W, R):
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return (U[:, :R] * S[:R].unsqueeze(0)) @ Vh[:R, :]

    with torch.no_grad():
        for R in ranks:
            # Matryoshka: use first R columns/rows of trained W_in/W_out
            if R >= HIDDEN:
                W_in_m_r = W_in_m
                W_out_m_r = W_out_m
            else:
                W_in_m_r = W_in_m[:, :R]
                W_out_m_r = W_out_m[:R, :]

            # For matryoshka, forward uses the truncated matrices directly (no SVD)
            def fwd_mat(h, R):
                x = h.float()
                if R >= HIDDEN:
                    return x + F.silu(x @ W_in_m.float()) @ W_out_m.float()
                return x + F.silu(x @ W_in_m[:, :R].float()) @ W_out_m[:R, :].float()

            # For SVD baseline, use SVD-truncated full-rank head
            if R >= HIDDEN:
                W_in_s_r = w_in_full
                W_out_s_r = w_out_full
            else:
                W_in_s_r = truncate_svd(w_in_full, R)
                W_out_s_r = truncate_svd(w_out_full, R)

            total_correct_m = 0; total_correct_s = 0; total = 0
            for h, vpred in zip(all_hidden, all_vpred):
                # Matryoshka
                x_m = fwd_mat(h, R)
                logits_m = x_m.to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
                pred_m = logits_m.float().argmax(dim=-1)
                # SVD
                x_s = head_forward(h, W_in_s_r, W_out_s_r)
                logits_s = x_s.to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
                pred_s = logits_s.float().argmax(dim=-1)

                valid = SEQ_LEN - 2
                total_correct_m += (pred_m[:valid] == vpred[:valid]).sum().item()
                total_correct_s += (pred_s[:valid] == vpred[:valid]).sum().item()
                total += valid

            acc_m = total_correct_m / total
            acc_s = total_correct_s / total
            improvement = acc_m / max(acc_s, 1e-6)
            print(f"{R:>6} {acc_m:>12.4f} {acc_s:>14.4f} {improvement:>11.2f}x")


if __name__ == "__main__":
    main()
