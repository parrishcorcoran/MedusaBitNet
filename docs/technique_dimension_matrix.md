# Technique × Dimension Matrix

*How each technique/aperture targets each dimension of the boundary-layer signal.*

Each row = a specific aperture (feature we can compute).
Each column = a dimension of the underlying boundary-layer signal.
Cells = how much of that dimension the aperture captures (✓✓ strong, ✓ modest, · weak/none).

## The 5 dimensions of boundary-layer signal (empirical, from our PCA)

- **Dim 1 — Sharpness**: how peaked is the next-token distribution RIGHT NOW
- **Dim 2 — Trajectory**: how has the distribution/state been evolving over recent history
- **Dim 3 — Structural**: syntactic/grammatical position cues
- **Dim 4 — Geometric**: where does current hidden state sit in the 14-dim manifold
- **Dim 5 — Cross-aperture**: do multiple independent measurements agree

## The matrix

| Aperture | D1 Sharpness | D2 Trajectory | D3 Structural | D4 Geometric | D5 Agreement | Cost (Medusa) | Cost (Zero-BB) | Relative contribution |
|---|---|---|---|---|---|---|---|---|
| **Softmax-derived** | | | | | | | | |
| content_conf | ✓✓ | · | · | · | · | 0 | 1 | 0.10 strongest existing |
| content_entropy | ✓✓ | · | · | · | · | 0 | 1 | 0.09 |
| logit_gap | ✓✓ | · | · | · | · | 0 | 1 | 0.08 |
| purity (Σp²) | ✓✓ | · | · | · | · | 0 | 1 | 0.05 redundant |
| top3_cov | ✓✓ | · | · | · | · | 0 | 1 | 0.08 |
| top10_cov | ✓ | · | · | · | · | 0 | 1 | 0.02 |
| softmax_skew | ✓ | · | · | · | · | 0 | 1 | 0.05 ★ |
| softmax_kurt | ✓ | · | · | · | · | 0 | 1 | 0.06 ★ |
| kl_trajectory | ✓ | ✓✓ | · | · | · | 0 | 1 | 0.08 ★ strong |
| d_entropy | ✓ | ✓✓ | · | · | · | 0 | 1 | 0.03 ★ |
| d2_entropy | ✓ | ✓✓ | · | · | · | 0 | 1 | 0.04 ★ |
| **Trajectory (past confidences)** | | | | | | | | |
| rc10 (rolling mean conf 10) | · | ✓✓ | · | · | · | 0 | 0 | 0.02 |
| rc50 (rolling mean conf 50) | · | ✓✓ | · | · | · | 0 | 0 | 0.02 |
| conf_deriv | · | ✓✓ | · | · | · | 0 | 0 | 0.02 |
| conf_lag1, conf_lag5 | · | ✓✓ | · | · | · | 0 | 0 | 0.02-0.03 |
| conf_var20 (rolling conf variance) | · | ✓✓ | · | · | · | 0 | 0 | 0.03 ★ |
| dir_persistence | · | ✓✓ | · | ✓ | · | 0 | 0 | 0.04 ★ |
| **Structural (token stream)** | | | | | | | | |
| dist_period_log | · | · | ✓✓ | · | · | 0 | 0 | 0.03 |
| dist_newline_log | · | · | ✓✓ | · | · | 0 | 0 | 0.02 |
| rel_pos | · | · | ✓✓ | · | · | 0 | 0 | 0.02 |
| bigram_freq | · | · | ✓✓ | · | · | 0 | 0 | 0.02 ★ |
| dist_same_log (token repetition) | · | · | ✓✓ | · | · | 0 | 0 | 0.04 ★ |
| vocab_div | · | · | ✓ | · | · | 0 | 0 | 0.02 ★ |
| sent_len_log | · | · | ✓ | · | · | 0 | 0 | 0.02 ★ |
| bracket_parity | · | · | ✓ | · | · | 0 | 0 | 0.01 ★ |
| **Geometric (hidden state)** | | | | | | | | |
| nbr_min_dist | · | · | · | ✓✓ | · | 0 | 1 | 0.04 ★ strong |
| nbr_mean_dist | · | · | · | ✓✓ | · | 0 | 1 | 0.06 ★ strongest geo |
| nbr_median_dist | · | · | · | ✓✓ | · | 0 | 1 | 0.04 ★ |
| cluster_mindist | · | · | · | ✓✓ | · | 0 | 1 | 0.09 ★ rivals content_conf |
| cluster_entropy | · | · | · | ✓✓ | · | 0 | 1 | 0.04 ★ |
| state_velocity | · | ✓ | · | ✓✓ | · | 0 | 1 | 0.05 ★ |
| state_accel | · | ✓ | · | ✓✓ | · | 0 | 1 | 0.03 ★ |
| hidden_norm | · | · | · | ✓✓ | · | 0 | 1 | 0.06 ★ |
| norm_drift | · | ✓ | · | ✓✓ | · | 0 | 1 | 0.04 ★ |
| cos_lag1, cos_lag5, cos_lag10, cos_lag50 | · | ✓ | · | ✓✓ | · | 0 | 1 | 0.05 ★ |
| local_id | · | · | · | ✓ | · | 0 | 1 | 0.02 ★ |
| local_spread | · | · | · | ✓✓ | · | 0 | 1 | 0.05 ★ |
| retrieval_entropy | · | · | · | ✓✓ | · | 0 | 1 | 0.03 ★ |
| retrieval_peak | · | · | · | ✓✓ | · | 0 | 1 | 0.03 ★ |
| fe_adjusted (free energy proxy) | · | · | · | ✓✓ | · | 0 | 1 | 0.05 ★ |
| **Cross-aperture (multi-head)** | | | | | | | | |
| agreement_count | · | · | · | · | ✓✓ | 3 | 4 | 0.05-0.08 |
| conf_var (across 4 heads) | · | · | · | · | ✓✓ | 3 | 4 | 0.02 |
| conf_min (across 4 heads) | · | · | · | · | ✓✓ | 3 | 4 | 0.02 |
| **Token-reuse (lexical H2O — user-suggested)** | | | | | | | | |
| token_cumcount | · | · | ✓✓ | · | · | 0 | 0 | 0.06 ★ very strong |
| token_freq_window | · | · | ✓✓ | · | · | 0 | 0 | 0.02 ★ |
| token_rank_window | · | · | ✓✓ | · | · | 0 | 0 | 0.02 ★ |
| is_heavy_hitter | · | · | ✓✓ | · | · | 0 | 0 | 0.02 ★ |
| distinct_in_window | · | · | ✓ | · | · | 0 | 0 | 0.02 ★ |
| **Spectral (dead or weak)** | | | | | | | | |
| fft_low/mid/high | · | ✓ | · | · | · | 0 | 1 | 0.00 dead |
| return_time_log | · | · | · | ✓ | · | 0 | 1 | 0.03 weak |
| pred_sensitivity | · | · | · | · | · | 1 | 2 | -0.01 hurts |
| pred_unique_count | · | · | · | · | ✓ | 3 | 4 | 0.00 |

★ = physics-framework-derived (came from the holographic/boundary/electron/etc. reasoning)

## Dimension coverage summary

| Dim | Apertures | Total relative contribution | Status |
|---|---|---|---|
| 1 Sharpness | 11 apertures | ~0.62 | Well-covered, diminishing returns |
| 2 Trajectory | 7 apertures | ~0.18 | Medium coverage |
| 3 Structural | 8 apertures | ~0.18 | Medium coverage |
| 4 Geometric | 15 apertures | ~0.60 | NEW, well-covered in cross-pattern |
| 5 Cross-aperture | 3 apertures | ~0.12 | Expensive (requires multi-head); could add cheap proxy |

## Apertures we haven't tested yet (infrastructure required)

| Aperture | Dim | Why interesting | Blocker |
|---|---|---|---|
| Attention-pattern similarity | D4/D5 | Orthogonal channel to hidden states | Need to extract attention weights from llama.cpp |
| Layer-wise hidden-state distance | D4 | Depth-of-stability signal | Need multi-layer cache |
| Gradient norm at hidden state | D1/D4 | Susceptibility analog | Need backward pass |
| Effective temperature | D1 | Thermodynamic analog | Redundant with entropy (probably) |
| KV-cache heavy-hitter pattern | D4 | Which past tokens matter? | Need attention extraction |
| Layer-entropy trajectory (each layer's predictive entropy) | D2/D4 | Info flow through depth | Need multi-layer + lm_head project per layer |

