"""Comprehensive held-out eval: compare all head variants at all ranks.

Variants:
  A) Original full-rank 4-head Medusa (head-0)
  B) F16 Matryoshka full (800 steps on 1000 seqs)
  C) Ternary Matryoshka pilot (300 steps on 5 seqs)
  D) SVD truncation of original full-rank head-0

Metric: head-0 accuracy vs vanilla greedy on seqs 36-47 (held out).
"""
import numpy as np
import torch
import torch.nn.functional as F

SEQ_LEN = 2048; HIDDEN = 2560


def weight_quant_ste(w):
    s = 1.0 / w.abs().mean().clamp_min(1e-5)
    return (w * s).round().clamp(-1, 1) / s


def truncate_svd(W, R):
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return (U[:, :R] * S[:R].unsqueeze(0)) @ Vh[:R, :]


def head_forward(hidden, W_in, W_out, rank=None, ternarize=False):
    x = hidden.float()
    Wi = W_in.float() if rank is None else W_in[:, :rank].float()
    Wo = W_out.float() if rank is None else W_out[:rank, :].float()
    if ternarize:
        Wi = weight_quant_ste(Wi)
        Wo = weight_quant_ste(Wo)
    return x + F.silu(x @ Wi) @ Wo


def main():
    # Load everything
    orig = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    W_in_orig = orig["heads"]["w_in"][0, 0].float()
    W_out_orig = orig["heads"]["w_out"][0, 0].float()

    mat = torch.load("checkpoints/matryoshka_full/head_step800.pt",
                     map_location="cpu", weights_only=True)
    W_in_mat = mat["W_in"].float()
    W_out_mat = mat["W_out"].float()

    tern = torch.load("checkpoints/ternary_full/head_step800.pt",
                      map_location="cpu", weights_only=True)
    W_in_tern = tern["W_in"].float()
    W_out_tern = tern["W_out"].float()

    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    test_seqs = list(range(36, 48))

    # Pre-load hiddens and vanilla greedy
    hs = []; vs = []
    for si in test_seqs:
        off = si * per_seq
        chunk = hidden_mm[off:off + per_seq]
        h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                   .view(SEQ_LEN, HIDDEN))
        vl = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
        vpred = vl.float().argmax(dim=-1)
        hs.append(h); vs.append(vpred)

    def eval_head(name, fwd_fn):
        correct = 0; total = 0
        with torch.no_grad():
            for h, vpred in zip(hs, vs):
                x_new = fwd_fn(h)
                logits = x_new.to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
                pred = logits.float().argmax(dim=-1)
                valid = SEQ_LEN - 2
                correct += (pred[:valid] == vpred[:valid]).sum().item()
                total += valid
        return correct / total

    ranks = [32, 64, 128, 256, 512, 1024, 2560]
    print(f"{'rank':>6} {'Original':>10} {'F16-Mat':>10} {'Ternary':>10} {'SVD-trunc':>10}")
    print("-" * 52)

    for R in ranks:
        # Original: full-rank only
        if R == 2560:
            acc_orig = eval_head(
                f"orig-{R}",
                lambda h: head_forward(h, W_in_orig, W_out_orig)
            )
        else:
            acc_orig = float('nan')  # N/A (original is full-rank only)

        # F16 Matryoshka
        acc_mat = eval_head(
            f"mat-{R}",
            lambda h, R=R: head_forward(h, W_in_mat, W_out_mat, rank=R if R < HIDDEN else None)
        )

        # Ternary Matryoshka (apply ternary quant at eval time too)
        acc_tern = eval_head(
            f"tern-{R}",
            lambda h, R=R: head_forward(h, W_in_tern, W_out_tern,
                                         rank=R if R < HIDDEN else None, ternarize=True)
        )

        # SVD truncation of original
        if R < HIDDEN:
            W_i_s = truncate_svd(W_in_orig, R)
            W_o_s = truncate_svd(W_out_orig, R)
        else:
            W_i_s, W_o_s = W_in_orig, W_out_orig
        acc_svd = eval_head(
            f"svd-{R}",
            lambda h, W_i=W_i_s, W_o=W_o_s: head_forward(h, W_i, W_o)
        )

        orig_str = f"{acc_orig:.4f}" if not np.isnan(acc_orig) else "     --"
        print(f"{R:>6} {orig_str:>10} {acc_mat:>10.4f} {acc_tern:>10.4f} {acc_svd:>10.4f}")


if __name__ == "__main__":
    main()
