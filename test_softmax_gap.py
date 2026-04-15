"""Softmax-gap (top-1 prob - top-2 prob) as feature vs logit-gap.

Gemini's suggestion: we currently use top1_logit - top2_logit (raw logits).
The softmax version (top1_prob - top2_prob) is a different aperture on
Dim 1 — potentially more calibrated since softmax is scale-normalized.
"""
import numpy as np
import torch
import torch.nn.functional as F
from model import MedusaHeads

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def main():
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN

    all_logit_gap = []; all_softmax_gap = []; all_conf = []; all_correct = []

    with torch.no_grad():
        for si in range(16):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            logits = heads(h.unsqueeze(0), lm_head)
            logits0 = logits[0, :, 0, :].float()
            probs0 = F.softmax(logits0, dim=-1)

            top2_l = torch.topk(logits0, 2, dim=-1).values
            top2_p = torch.topk(probs0, 2, dim=-1).values
            logit_gap = (top2_l[:, 0] - top2_l[:, 1]).numpy()
            softmax_gap = (top2_p[:, 0] - top2_p[:, 1]).numpy()
            conf = probs0.max(dim=-1).values.numpy()
            pred = probs0.argmax(dim=-1).numpy()

            vlogits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vpred = vlogits.float().argmax(dim=-1).numpy()
            valid = SEQ_LEN - 2
            correct = (pred[:valid] == vpred[:valid]).astype(np.int8)

            all_logit_gap.append(logit_gap[:valid])
            all_softmax_gap.append(softmax_gap[:valid])
            all_conf.append(conf[:valid])
            all_correct.append(correct)

    logit_gap = np.concatenate(all_logit_gap)
    softmax_gap = np.concatenate(all_softmax_gap)
    conf = np.concatenate(all_conf)
    correct = np.concatenate(all_correct).astype(np.float32)
    N = len(correct)
    print(f"positions: {N:,}")

    # Correlations
    print(f"\nCorrelations:")
    print(f"  logit_gap   vs correct: {np.corrcoef(logit_gap, correct)[0,1]:.4f}")
    print(f"  softmax_gap vs correct: {np.corrcoef(softmax_gap, correct)[0,1]:.4f}")
    print(f"  conf        vs correct: {np.corrcoef(conf, correct)[0,1]:.4f}")
    print(f"  logit_gap   vs softmax_gap: {np.corrcoef(logit_gap, softmax_gap)[0,1]:.4f}")
    print(f"  conf        vs softmax_gap: {np.corrcoef(conf, softmax_gap)[0,1]:.4f}")

    # Single-feature gate frontier each
    def frontier_(score, label, targets):
        order = np.argsort(-score)
        sorted_c = label[order]
        cum = np.cumsum(sorted_c)
        counts = np.arange(1, len(sorted_c) + 1)
        fid = cum / counts
        results = []
        for λ in targets:
            ok = fid >= λ
            if not ok.any():
                results.append((λ, 0.0, 0.0)); continue
            largest = np.where(ok)[0][-1]
            results.append((λ, (largest + 1) / len(sorted_c), fid[largest]))
        return results

    targets = [0.85, 0.90, 0.95, 0.99]
    print(f"\nSingle-feature frontier (skip rate at target fidelity):")
    print(f"{'feature':>16} {'λ=0.85':>10} {'λ=0.90':>10} {'λ=0.95':>10} {'λ=0.99':>10}")
    for name, score in [("conf", conf), ("logit_gap", logit_gap), ("softmax_gap", softmax_gap)]:
        fr = frontier_(score, correct, targets)
        print(f"{name:>16}  {fr[0][1]:>9.4f} {fr[1][1]:>9.4f} {fr[2][1]:>9.4f} {fr[3][1]:>9.4f}")


if __name__ == "__main__":
    main()
