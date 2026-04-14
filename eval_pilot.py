"""Evaluate pilot head accuracy on held-out sequences."""
import numpy as np
import torch
from model import MedusaHeads
from dataset import PackedTokenDataset

SEQ_LEN = 2048
HIDDEN  = 2560
DEVICE  = "cpu"

def main():
    ckpt = torch.load("checkpoints/pilot_gguf/medusa_heads_step500.pt",
                      map_location=DEVICE, weights_only=True)
    cfg = ckpt.get("cfg", {}) or {}
    if not isinstance(cfg, dict):
        cfg = vars(cfg)
    n_heads = cfg.get("num_heads", 1)
    n_layers = cfg.get("num_layers_per_head", 1)
    hidden_size = HIDDEN
    vocab_size = 128256
    print(f"ckpt cfg: n_heads={n_heads} layers={n_layers} hidden={hidden_size} vocab={vocab_size}")

    heads = MedusaHeads(hidden_size, vocab_size, n_heads, n_layers,
                        dtype=torch.bfloat16).to(DEVICE)
    heads.load_state_dict(ckpt["heads"])
    heads.eval()

    lm_head = torch.load("data/lm_head.pt", map_location=DEVICE,
                         weights_only=True).to(torch.bfloat16)  # [V, H]

    hidden_bin = np.memmap("data/hidden_gguf_v2_holdout.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    tokens = PackedTokenDataset("data/tokens.bin", SEQ_LEN)

    holdout_token_idxs = [5, 6]

    with torch.no_grad():
        for si, tok_idx in enumerate(holdout_token_idxs):
            start = si * per_seq
            chunk = hidden_bin[start:start + per_seq]
            h = (torch.from_numpy(chunk.copy())
                      .view(torch.bfloat16).view(SEQ_LEN, HIDDEN).to(DEVICE))
            h_b = h.unsqueeze(0)  # [1, T, H]

            targets = tokens[tok_idx]  # [SEQ_LEN+1]
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            # Head k (0-indexed) predicts t+1+k. For a single head, that's t+1.
            gold_t1 = targets[1:SEQ_LEN+1].long()

            logits = heads(h_b, lm_head)  # [1, T, k, V]
            preds = logits.argmax(-1)  # [1, T, k]
            # Head 0 is predicting position i -> token at i+1
            pred_h0 = preds[0, :, 0]
            acc = (pred_h0 == gold_t1).float().mean().item()
            print(f"  holdout seq tok_idx={tok_idx}  acc@1 (head-0 -> t+1) = {acc:.4f}")

if __name__ == "__main__":
    main()
