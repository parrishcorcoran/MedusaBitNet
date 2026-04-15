"""Offline confidence-gate test on cached GGUF hidden states.

Validates the core thesis claim: head-0 softmax peak IS a usable gate signal.
Measures the efficiency frontier (fraction of tokens skippable vs fidelity)
without needing to actually modify any decode loop.

For each position t in cached hidden states:
    1. Predict: head-0(hidden[t]) -> p_head (distribution over vocab)
    2. Ground truth: argmax(lm_head @ hidden[t+2]) = what vanilla greedy emits at t+2
       (head trained with shift=k+2, so head-0 predicts the token two ahead)
    3. Gate: would we skip if peak(p_head) > tau?
    4. Correct: does argmax(p_head) == ground_truth?

Summary:
    for each tau:
        skip_rate = fraction of tokens passing gate
        fidelity  = among passing, fraction where head matches vanilla
        effective = skip_rate * fidelity (tokens correctly skippable)
"""
import numpy as np
import torch
import torch.nn.functional as F
from model import MedusaHeads

SEQ_LEN = 2048
HIDDEN  = 2560
VOCAB   = 128256


def main():
    # Load heads
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    cfg = ckpt.get("cfg", {}) or {}
    if not isinstance(cfg, dict): cfg = vars(cfg)
    num_heads = cfg.get("num_heads", 4)
    num_layers = cfg.get("num_layers_per_head", 1)
    heads = MedusaHeads(HIDDEN, VOCAB, num_heads, num_layers,
                        dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    print(f"[gate] heads loaded: K={num_heads} L={num_layers}")

    # Load lm_head
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)  # [V, H]
    print(f"[gate] lm_head: {tuple(lm_head.shape)}")

    # Load hidden cache (mmap so we don't eat 10GB of RAM)
    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    n_seqs = hidden_mm.size // per_seq
    print(f"[gate] hidden cache: {n_seqs} seqs x {SEQ_LEN} tokens")

    # Sample a few test sequences, score each
    n_test_seqs = min(16, n_seqs)
    # Use positions 0..SEQ_LEN-3 (need t+2 within range)

    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for si in range(n_test_seqs):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            h_b = h.unsqueeze(0)  # [1, T, H]

            # Medusa heads in batch
            logits_all = heads(h_b, lm_head)  # [1, T, K, V]

            # Head 0 prediction and confidence for each position
            head0_logits = logits_all[0, :, 0, :]  # [T, V]
            # positions 0..T-3 have valid t+2 target
            valid = SEQ_LEN - 2
            head0_logits = head0_logits[:valid]
            head0_probs = F.softmax(head0_logits.float(), dim=-1)
            confidence = head0_probs.max(dim=-1).values  # [valid]
            head0_pred = head0_probs.argmax(dim=-1)      # [valid]

            # Vanilla greedy at position t+2 = argmax(lm_head @ hidden[t+1]).
            # (hidden[t+1] encodes "what comes next" given context through t+1,
            #  which IS the token at t+2.) Head-0 with shift=k+2 predicts the
            # token at t+2 from hidden[t], so these are comparable.
            vanilla_logits = h[1:1 + valid].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vanilla_pred = vanilla_logits.float().argmax(dim=-1)  # [valid]

            correct = (head0_pred == vanilla_pred).float()  # [valid]

            all_confidences.append(confidence.numpy())
            all_correct.append(correct.numpy())
            if si < 3:
                print(f"  seq {si}: head0 overall acc@1 = {correct.mean().item():.4f}, "
                      f"mean confidence = {confidence.mean().item():.4f}")

    all_conf = np.concatenate(all_confidences)
    all_corr = np.concatenate(all_correct)
    print(f"\n[gate] total positions scored: {len(all_conf):,}")
    print(f"[gate] overall head-0 acc@1: {all_corr.mean():.4f}")
    print(f"[gate] confidence stats: mean={all_conf.mean():.4f} median={np.median(all_conf):.4f}")

    # Efficiency frontier: for each tau, report skip rate and fidelity
    print(f"\n{'tau':>6} {'skip_rate':>10} {'fidelity':>10} {'effective':>10} {'naive_speedup':>14}")
    print("-" * 60)
    for tau in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995]:
        mask = all_conf > tau
        skip_rate = mask.mean()
        if mask.sum() == 0:
            fidelity = float('nan')
        else:
            fidelity = all_corr[mask].mean()
        effective = skip_rate * fidelity if not np.isnan(fidelity) else 0
        # Naive speedup: if we skip fraction `skip_rate` with accuracy `fidelity`,
        # and we only trust skips that are correct, effective skip = effective.
        # Compute ceiling: 1 / (1 - effective) assuming skipped tokens cost 0
        naive_speedup = 1.0 / max(1.0 - effective, 1e-9)
        print(f"{tau:>6.3f} {skip_rate:>10.4f} {fidelity:>10.4f} {effective:>10.4f} {naive_speedup:>14.2f}x")

    # Also report calibration: buckets of confidence
    print(f"\nCalibration (buckets of head-0 softmax peak):")
    print(f"{'bucket':>20} {'n':>8} {'mean_conf':>10} {'accuracy':>10}")
    edges = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (all_conf >= lo) & (all_conf < hi)
        if m.sum() == 0: continue
        print(f"{str(f'[{lo:.2f},{hi:.2f})'):>20} {int(m.sum()):>8} "
              f"{all_conf[m].mean():>10.4f} {all_corr[m].mean():>10.4f}")


if __name__ == "__main__":
    main()
