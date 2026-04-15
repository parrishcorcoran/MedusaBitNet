"""Gate test varying the SUBSET of heads used for ensemble agreement.

Hypothesis: head-3 is weak (12% acc) and drags down ensemble skip rate.
Smaller subsets (heads 0-1 or 0-1-2) should find a better fidelity/skip
trade-off.
"""
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
from model import MedusaHeads

SEQ_LEN = 2048
HIDDEN  = 2560
VOCAB   = 128256


def main():
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    n_test_seqs = 16

    all_preds = []   # [K, N]
    all_confs = []
    all_vanilla = []

    with torch.no_grad():
        for si in range(n_test_seqs):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            logits_all = heads(h.unsqueeze(0), lm_head)
            probs_all = F.softmax(logits_all.float(), dim=-1)
            preds_all = probs_all.argmax(dim=-1)[0]
            confs_all = probs_all.max(dim=-1).values[0]

            vanilla_logits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vanilla_all = vanilla_logits.float().argmax(dim=-1)

            t_start, t_end = 3, SEQ_LEN - 2
            ts = torch.arange(t_start, t_end)
            vanilla = vanilla_all[ts]

            headk_preds = torch.stack([preds_all[ts - k, k] for k in range(4)], dim=0)
            headk_confs = torch.stack([confs_all[ts - k, k] for k in range(4)], dim=0)
            all_preds.append(headk_preds.numpy())
            all_confs.append(headk_confs.numpy())
            all_vanilla.append(vanilla.numpy())

    preds = np.concatenate(all_preds, axis=1)  # [4, N]
    confs = np.concatenate(all_confs, axis=1)  # [4, N]
    vanilla = np.concatenate(all_vanilla)
    N = preds.shape[1]
    print(f"Total positions: {N:,}\n")

    head_sets = [(0,), (0,1), (0,1,2), (0,1,2,3), (0,2), (0,3)]
    for hs in head_sets:
        print(f"=== Heads {hs} (strict agreement + all conf > tau) ===")
        print(f"{'tau':>6} {'skip':>8} {'fidelity':>10} {'effective':>10} {'speedup':>10}")
        print("-" * 50)
        for tau in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            # All heads in set predict same token
            first_pred = preds[hs[0]]
            agree = np.all(np.stack([preds[k] == first_pred for k in hs]), axis=0)
            # All heads in set have conf > tau
            conf_ok = np.all(np.stack([confs[k] > tau for k in hs]), axis=0)
            m = agree & conf_ok
            sk = m.mean()
            fid = (first_pred[m] == vanilla[m]).mean() if m.sum() else float('nan')
            eff = sk * fid if not np.isnan(fid) else 0
            sp = 1.0 / max(1.0 - eff, 1e-9)
            print(f"{tau:>6.3f} {sk:>8.4f} {fid:>10.4f} {eff:>10.4f} {sp:>9.2f}x")
        print()


if __name__ == "__main__":
    main()
