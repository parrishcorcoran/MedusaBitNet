"""Multi-head ensemble gate: require all K heads to agree AND all to be confident.

Hypothesis: each head is a noisy sensor of the same underlying entropy/cloud
signal. Requiring consensus filters the 'confidently wrong' failures that
any single head exhibits, at the cost of fewer gated tokens (lower skip rate
but higher fidelity).

For each position t:
  - head_k predicts token at t+k+2 (trained shift=k+2)
  - So we can only check agreement on "what is t+2" via head_0 directly, or
    use head_k at position (t - k) to also predict t+2.

This version uses the SAME anchor t: measures whether head-0's top-1 at pos t,
head-1's top-1 at pos t-1, head-2's top-1 at pos t-2, head-3's top-1 at pos t-3
all agree on the token at position t+2. That's a 4-way consensus on the same
future position from four different hidden states — closer to the real
speculation chain llama-medusa does.

Each head also reports its own confidence at its own anchor. Gate fires iff
all four max-probs exceed tau.

Fidelity: fraction of consensus-gated predictions that match vanilla greedy
(argmax(lm_head @ hidden[t+1])).
"""
import numpy as np
import torch
import torch.nn.functional as F
from model import MedusaHeads

SEQ_LEN = 2048
HIDDEN  = 2560
VOCAB   = 128256


def main():
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    num_heads, num_layers = 4, 1
    heads = MedusaHeads(HIDDEN, VOCAB, num_heads, num_layers,
                        dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])

    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    n_seqs = hidden_mm.size // per_seq

    n_test_seqs = 16
    # Valid range: need positions t-3 through t+1 to exist.
    # So t ∈ [3, SEQ_LEN - 2), and we'll track predictions for position t+2
    # ∈ [5, SEQ_LEN).

    all_head0_conf = []   # head-0 confidence at anchor t
    all_head0_pred = []
    all_headk_preds = []  # [K, valid] head-k top-1 for same target t+2
    all_headk_confs = []  # [K, valid] head-k confidence at anchor t-k
    all_vanilla = []
    all_single_correct = []

    with torch.no_grad():
        for si in range(n_test_seqs):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            h_b = h.unsqueeze(0)
            logits_all = heads(h_b, lm_head)  # [1, T, K, V]
            probs_all = F.softmax(logits_all.float(), dim=-1)  # [1, T, K, V]
            preds_all = probs_all.argmax(dim=-1)[0]            # [T, K]
            confs_all = probs_all.max(dim=-1).values[0]        # [T, K]

            # Vanilla greedy for token at position p via argmax(lm_head @ hidden[p-1]).
            # vanilla_logits[j] = h[j+1] @ lm_head.T predicts the token at
            # position (j+1)+1 = j+2. So vanilla_all[j] is greedy at position j+2.
            vanilla_logits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vanilla_all = vanilla_logits.float().argmax(dim=-1)  # vanilla_all[j] -> position j+2

            # Target positions p = t+2, where t is anchor for head-0.
            # Each head k uses its own anchor t-k, to predict same p = t+2.
            # Need t-k >= 0, so t >= K-1 = 3. And p = t+2 <= T-1 (max vanilla idx).
            # vanilla_all has length T-1, indexed by position minus 1. So
            # vanilla at position p corresponds to vanilla_all[p-1].
            t_start = num_heads - 1  # 3
            t_end = SEQ_LEN - 2      # exclusive; need t+2 <= T-1 so t <= T-3
            valid = t_end - t_start
            ts = torch.arange(t_start, t_end)

            # Head k at anchor t-k predicts position t+2 (since its shift = k+2,
            # so from position (t-k) it predicts (t-k) + (k+2) = t+2).
            headk_preds = torch.stack([preds_all[ts - k, k] for k in range(num_heads)], dim=0)  # [K, valid]
            headk_confs = torch.stack([confs_all[ts - k, k] for k in range(num_heads)], dim=0)  # [K, valid]

            # Vanilla at target position p = t+2. Since vanilla_all[j] indexes
            # position j+2, we need vanilla_all[t].
            vanilla = vanilla_all[ts]  # [valid]

            head0_correct = (headk_preds[0] == vanilla)

            all_head0_conf.append(headk_confs[0].numpy())
            all_head0_pred.append(headk_preds[0].numpy())
            all_headk_preds.append(headk_preds.numpy())
            all_headk_confs.append(headk_confs.numpy())
            all_vanilla.append(vanilla.numpy())
            all_single_correct.append(head0_correct.numpy().astype(np.float32))

    head0_conf = np.concatenate(all_head0_conf)
    head0_pred = np.concatenate(all_head0_pred)
    headk_preds = np.concatenate(all_headk_preds, axis=1)  # [K, N]
    headk_confs = np.concatenate(all_headk_confs, axis=1)  # [K, N]
    vanilla = np.concatenate(all_vanilla)
    correct = np.concatenate(all_single_correct)

    N = len(head0_conf)
    print(f"[ensemble] scored positions: {N:,}")
    print(f"[ensemble] head-0 overall acc@1: {correct.mean():.4f}")

    # Single-head gate (baseline)
    print(f"\n=== Single-head (head-0 only) gate ===")
    print(f"{'tau':>6} {'skip':>8} {'fidelity':>10} {'effective':>10} {'speedup':>10}")
    print("-" * 50)
    for tau in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        m = head0_conf > tau
        sk = m.mean()
        fid = (head0_pred[m] == vanilla[m]).mean() if m.sum() else float('nan')
        eff = sk * fid if not np.isnan(fid) else 0
        sp = 1.0 / max(1.0 - eff, 1e-9)
        print(f"{tau:>6.3f} {sk:>8.4f} {fid:>10.4f} {eff:>10.4f} {sp:>9.2f}x")

    # Ensemble: all K heads agree AND all confident > tau
    print(f"\n=== Ensemble: all {num_heads} heads agree AND all conf > tau ===")
    print(f"{'tau':>6} {'skip':>8} {'fidelity':>10} {'effective':>10} {'speedup':>10}")
    print("-" * 50)
    all_agree = np.all(headk_preds == headk_preds[0:1], axis=0)  # [N] bool
    min_conf = headk_confs.min(axis=0)  # [N]
    for tau in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        m = all_agree & (min_conf > tau)
        sk = m.mean()
        fid = (headk_preds[0][m] == vanilla[m]).mean() if m.sum() else float('nan')
        eff = sk * fid if not np.isnan(fid) else 0
        sp = 1.0 / max(1.0 - eff, 1e-9)
        print(f"{tau:>6.3f} {sk:>8.4f} {fid:>10.4f} {eff:>10.4f} {sp:>9.2f}x")

    # Softer ensemble: k-of-K heads agree + head-0 confident > tau
    for agree_k in [2, 3]:
        print(f"\n=== Agree {agree_k}-of-{num_heads} + head-0 conf > tau ===")
        print(f"{'tau':>6} {'skip':>8} {'fidelity':>10} {'effective':>10} {'speedup':>10}")
        print("-" * 50)
        # For each pos, count how many heads predict the same as head-0
        agree_count = (headk_preds == headk_preds[0:1]).sum(axis=0)  # [N], 1..K
        for tau in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            m = (agree_count >= agree_k) & (head0_conf > tau)
            sk = m.mean()
            fid = (headk_preds[0][m] == vanilla[m]).mean() if m.sum() else float('nan')
            eff = sk * fid if not np.isnan(fid) else 0
            sp = 1.0 / max(1.0 - eff, 1e-9)
            print(f"{tau:>6.3f} {sk:>8.4f} {fid:>10.4f} {eff:>10.4f} {sp:>9.2f}x")


if __name__ == "__main__":
    main()
