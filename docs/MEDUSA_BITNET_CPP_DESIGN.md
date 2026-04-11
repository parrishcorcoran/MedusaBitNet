# Medusa Tree-Speculative Decoding in bitnet.cpp — Design Spec

*Integration work by Parrish Corcoran — first end-to-end combination of Medusa
speculative decoding with Microsoft's ternary-weight BitNet b1.58 inference
engine.*

**Research goal.** Integrate Medusa-style tree-speculative decoding into Microsoft's
`bitnet.cpp` (a fork of `llama.cpp` with custom ternary-weight GEMM kernels), so that a
single BitNet b1.58 model can self-speculate across `k` parallel Medusa heads and
verify `N ≫ k` candidate continuations in one backbone forward pass. Combined with
BitNet's LUT-based ternary GEMM, this targets peak theoretical throughput for
sub-bit-level LLM inference on commodity CPUs: `bitnet.cpp speed × Medusa 2-3×
acceptance`.

This document is the exhaustive implementation spec. Every C++ edit is listed with
file, line, and exact purpose. Tensor names, shapes, and data layouts are fixed so
that the Python training pipeline (`MedusaBitNet/train.py`), the weight converter
(`tools/convert_medusa_heads.py`), and the C++ runtime all agree.

All line numbers refer to
`/home/supercomputerz8/bitnet.cpp/3rdparty/llama.cpp/src/llama.cpp` unless stated
otherwise.

---

## 1. Architecture recap

```
                    ┌──────────────────────────────┐
                    │     BitNet b1.58 backbone    │
 tokens ──►────────►│  (ternary LUT GEMM kernels)  │────► h_T  (final hidden state)
                    └──────────────────────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                  ┌──────▼──────┐       ┌──────▼──────┐   (k = 4 heads)
                  │ Medusa head │  ...  │ Medusa head │
                  │     0       │       │     k-1     │
                  └──────┬──────┘       └──────┬──────┘
                         │ tied LM head        │
                         ▼                     ▼
                    logits_t+1            logits_t+k
```

Each head is a residual block: `h' = h + W_out · SiLU(W_in · h)` with `W_in`, `W_out ∈
ℝ^{H×H}` (H = backbone hidden size, 2560 for BitNet-2B). The final vocab projection
reuses the backbone's tied token embedding matrix (`model.tok_embd`). With
`num_layers_per_head = 1` the per-head parameter count is `2·H²` bf16.

Generation algorithm:

1. **Prefill** context with the normal backbone forward. `h_T = backbone(prompt)`.
2. **Draft.** Head 0 samples token `t₁` at position T+1, head 1 samples `t₂` at T+2,
   ..., head k-1 samples `t_k` at T+k. Each head also proposes its top-M candidates
   — these are the branching factor of the Medusa tree.
3. **Build tree.** Assemble a sparse tree of candidate continuations using the
   Medusa-published "64-node" tree topology (or any fixed sparse tree). Every tree
   node has a unique (position, parent) tuple.
4. **Verify.** One backbone forward pass over all tree-node tokens with a **tree
   attention mask** that lets each node attend only to its ancestors (not siblings).
   Extract the backbone's own-next-token distribution at every node.
5. **Accept.** Walk the tree from the root. At each node, if the backbone's argmax
   matches the speculated token, advance; otherwise stop. The longest matched prefix
   is accepted in a single step.
6. **Commit KV cache.** Keep only the KV entries on the accepted path; discard all
   rejected tree branches using `llama_kv_cache_seq_rm`.
7. **Repeat** from step 2 using the new accepted prefix.

With 64 tree nodes and a trained acceptance rate of ~0.6-0.7 per level, Medusa
typically accepts ~2.5 tokens per verify pass. Combined with bitnet.cpp's ~10-20
tok/s baseline this targets **25-50 tok/s on BitNet-2B**, which would exceed any
published CPU number.

---

## 2. Tensor name and shape convention

Used by `convert_medusa_heads.py` (writer) and the C++ loader (reader). Both sides
must agree exactly.

### Medusa tensors

| Name                                                 | Shape       | Dtype  | Notes                                    |
| ---------------------------------------------------- | ----------- | ------ | ---------------------------------------- |
| `medusa.head.{k}.layer.{l}.w_in.weight`              | `[H, H]`    | F16    | `k ∈ [0, n_heads)`, `l ∈ [0, n_layers)`  |
| `medusa.head.{k}.layer.{l}.w_out.weight`             | `[H, H]`    | F16    | zero-initialised at train start          |

Shape convention: GGUF stores `[n_cols, n_rows]` (first dim is K/input, second is
M/output in `ggml_mul_mat`). Since Medusa blocks are square (`H × H`), storage order
matches the torch `[in, out]` convention used by `nn.Linear.weight.T`. The
converter produces exactly what `ggml_mul_mat(W, x)` expects.

### Medusa metadata

| Key                          | Type | Value                                         |
| ---------------------------- | ---- | --------------------------------------------- |
| `medusa.n_heads`             | u32  | number of Medusa heads (k)                    |
| `medusa.n_layers_per_head`   | u32  | residual blocks per head (usually 1)          |
| `medusa.hidden_size`         | u32  | backbone hidden size (sanity check against `llama.embedding_length`) |

The C++ loader treats all Medusa tensors and metadata as **optional**: a vanilla
BitNet GGUF still loads and runs normally. A Medusa-enabled GGUF has the extra
tensors and the runtime detects them via `ctx->model->n_medusa_heads > 0`.

---

## 3. C++ edits — exhaustive list

All edits live in the `3rdparty/llama.cpp` submodule. We'll keep them on a branch
of the submodule (`medusa-integration`) so upstream rebases stay tractable. No
changes to the top-level `bitnet.cpp` repo are needed for the runtime; we'll add
a new `examples/medusa/` driver and a thin Python wrapper later.

### 3.1 `include/llama.h` — public API additions

**Location:** header, near the end of the public API block.

Add four new public functions:

```c
// Return the number of Medusa heads loaded with this model (0 if none).
LLAMA_API int32_t llama_n_medusa_heads(const struct llama_model * model);

// Return the number of residual layers per Medusa head.
LLAMA_API int32_t llama_n_medusa_layers(const struct llama_model * model);

// After llama_decode, return a pointer to logits produced by Medusa head `k`
// for the token at batch index `i`. Layout mirrors llama_get_logits_ith:
// returns NULL if Medusa is not loaded or indices are out of range.
LLAMA_API float * llama_get_medusa_logits_ith(
        struct llama_context * ctx,
        int32_t                k,
        int32_t                i);

// Enable/disable tree attention. When enabled, llama_set_inputs consults
// ctx->medusa_tree to populate inp_KQ_mask instead of the default causal mask.
LLAMA_API void llama_set_tree_attention(
        struct llama_context *           ctx,
        const struct llama_medusa_tree * tree); // NULL disables
```

And a new public struct describing the tree mask:

```c
// A sparse Medusa speculation tree. The caller owns the storage.
// n_nodes:       total tokens in the batch (must equal batch.n_tokens)
// parent[i]:     index of node i's parent, or -1 for the root.
//                Node 0 is always the root and corresponds to the first batch
//                token; all other nodes are speculative children.
// depth[i]:      distance from the root (root has depth 0); used to compute
//                absolute token positions.
struct llama_medusa_tree {
    int32_t         n_nodes;
    const int32_t * parent;
    const int32_t * depth;
};
```

The tree mask is therefore `[n_kv, n_tokens]` where entry `(kv_pos, node_i)` is
`0.0f` if `kv_pos` belongs to `node_i`'s root→node_i ancestor path, and
`-INFINITY` otherwise. We build this in `llama_set_inputs` (§3.5).

### 3.2 `src/llama.cpp` — `llama_model` struct (around line 2941)

After the existing `output` / `output_b` fields, add:

```cpp
    // ---- Medusa heads (optional) ----
    // Sized n_medusa_heads * n_medusa_layers_per_head. Indexed as
    // [k * n_layers_per_head + l].
    int32_t n_medusa_heads          = 0;
    int32_t n_medusa_layers_per_head = 0;
    std::vector<struct ggml_tensor *> medusa_w_in;
    std::vector<struct ggml_tensor *> medusa_w_out;
```

These are zero-default so vanilla BitNet models behave exactly as before.

### 3.3 `src/llama.cpp` — model loader for `LLM_ARCH_BITNET_B158` (after line 8731)

Inside the `case LLM_ARCH_BITNET_B158:` branch, after loading `model.output`
(currently line 8728), insert optional Medusa loading **before** the layer loop
at line 8733:

```cpp
// ---- optional Medusa heads ----
{
    uint32_t n_heads_kv  = 0;
    uint32_t n_layers_kv = 0;
    const bool ok_heads  = ml.get_key("medusa.n_heads",           n_heads_kv,  false);
    const bool ok_layers = ml.get_key("medusa.n_layers_per_head", n_layers_kv, false);
    if (ok_heads && ok_layers && n_heads_kv > 0) {
        model.n_medusa_heads           = (int32_t) n_heads_kv;
        model.n_medusa_layers_per_head = (int32_t) n_layers_kv;

        const int n = model.n_medusa_heads * model.n_medusa_layers_per_head;
        model.medusa_w_in .resize(n, nullptr);
        model.medusa_w_out.resize(n, nullptr);

        for (int k = 0; k < model.n_medusa_heads; ++k) {
            for (int l = 0; l < model.n_medusa_layers_per_head; ++l) {
                char buf_in[128], buf_out[128];
                snprintf(buf_in,  sizeof(buf_in),
                         "medusa.head.%d.layer.%d.w_in.weight", k, l);
                snprintf(buf_out, sizeof(buf_out),
                         "medusa.head.%d.layer.%d.w_out.weight", k, l);
                const int idx = k * model.n_medusa_layers_per_head + l;
                model.medusa_w_in [idx] = ml.create_tensor(
                    ctx_output_split, buf_in,  {n_embd, n_embd});
                model.medusa_w_out[idx] = ml.create_tensor(
                    ctx_output_split, buf_out, {n_embd, n_embd});
            }
        }
        LLAMA_LOG_INFO("%s: loaded %d Medusa heads, %d layers each\n",
                       __func__, model.n_medusa_heads,
                       model.n_medusa_layers_per_head);
    }
}
```

The `false` argument to `get_key` makes the metadata optional. `create_tensor`
here is required (non-optional) — if the metadata is present but the tensors are
missing, loading should fail loudly.

### 3.4 `src/llama.cpp` — graph build, `build_bitnet_158` (insert at line 15527)

The final hidden state `cur` becomes available right after `result_norm` at line
15527. Modify the block from line 15527 through 15534 to:

```cpp
cb(cur, "result_norm", -1);

// Tap the post-norm hidden state for Medusa before the main LM head projection.
struct ggml_tensor * h_final = cur;

// ---- main LM head (unchanged from upstream) ----
cur = llm_build_lora_mm(lctx, ctx0, model.tok_embd, cur);
cb(cur, "result_output", -1);
ggml_build_forward_expand(gf, cur);

// ---- optional Medusa heads ----
if (model.n_medusa_heads > 0) {
    const int K = model.n_medusa_heads;
    const int L = model.n_medusa_layers_per_head;
    for (int k = 0; k < K; ++k) {
        struct ggml_tensor * h = h_final;          // [n_embd, n_outputs]
        for (int l = 0; l < L; ++l) {
            const int idx = k * L + l;
            // z = SiLU(W_in @ h)
            struct ggml_tensor * z = ggml_mul_mat(ctx0, model.medusa_w_in[idx], h);
            z = ggml_silu(ctx0, z);
            // out = W_out @ z
            z = ggml_mul_mat(ctx0, model.medusa_w_out[idx], z);
            // residual
            h = ggml_add(ctx0, h, z);
        }
        // Tied LM head projection.
        struct ggml_tensor * medusa_logits = llm_build_lora_mm(
            lctx, ctx0, model.tok_embd, h);
        // Name each one uniquely so we can retrieve them from the compute graph.
        char name[64];
        snprintf(name, sizeof(name), "result_medusa_%d", k);
        cb(medusa_logits, name, -1);
        ggml_build_forward_expand(gf, medusa_logits);
    }
}

return gf;
```

Key points:

* We use `ggml_mul_mat` directly (not `llm_build_lora_mm`) for the Medusa
  internal matmuls because they have no LoRA adapters attached and the input is
  already in F16/F32, not BitNet-packed.
* The final projection to vocab reuses `llm_build_lora_mm(..., tok_embd, ...)`
  just like the main LM head, so the Medusa outputs share the backbone's LoRA
  pathway if any.
* `h_final` is `[n_embd, n_outputs]` — it has already been filtered through
  `ggml_get_rows(cur, inp_out_ids)` in the last-layer block at line 15484. That
  means Medusa logits are also `[n_vocab, n_outputs]`, matching the main logits
  layout exactly.
* Each head's output tensor gets a unique callback name (`result_medusa_0`,
  `..._1`, ...) which is how `llama_decode_internal` will fish them out of the
  graph after compute.

### 3.5 `src/llama.cpp` — `llama_set_inputs`, tree attention mask

**UPDATE (post-implementation finding):** §3.5 is **not required**. The
existing causal mask fill loop in `llama_set_inputs` (lines 17100-17174)
already implements tree attention correctly when the batch is filled with
proper seq_id inheritance: a tree node that's an ancestor of multiple
branches must carry ALL those branches' seq_ids in its
`batch.seq_id[i][...]` array, and its KV cell will inherit them. The
default rule `has_seq_id(q_seq) && cell.pos <= q_pos` then gives exact
tree visibility without any custom fill.

This matches exactly what `examples/speculative/speculative.cpp` does when
it branches (the "propagate seq_id to prior batch tokens" loop in its
drafting code). We'll do the same in the Medusa driver (§3.7).

The `lctx.medusa_tree` field and `llama_set_tree_attention` API are still
kept — they're harmless hooks for future dynamic-tree extensions — but no
C++ changes to `llama_set_inputs` are needed.

**Original (superseded) design below, kept for reference:**

Insert a new branch **before** the existing causal-mask fill loop. Field
`lctx.medusa_tree` (a `const llama_medusa_tree *`, NULL by default) is added to
`struct llama_context`.

```cpp
if (lctx.medusa_tree && lctx.inp_KQ_mask) {
    GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));
    GGML_ASSERT(lctx.medusa_tree->n_nodes == (int32_t) batch.n_tokens);

    float * data = (float *) lctx.inp_KQ_mask->data;
    const int32_t n_kv     = kv_self.n;
    const int32_t n_tokens = batch.n_tokens;

    // Mark every entry masked by default.
    const int64_t stride_row = n_kv;              // per query token
    const int64_t padded_q   = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
    std::fill_n(data, padded_q * n_kv, -INFINITY);

    // For each tree node, walk up to the root and mark each ancestor's KV
    // cell as visible. The ancestor's KV cell is found by (seq_id, pos).
    for (int32_t q = 0; q < n_tokens; ++q) {
        const llama_pos    q_pos    = batch.pos[q];
        const llama_seq_id q_seq    = batch.seq_id[q][0];

        // 1) all historical context (KV cells with pos < node's root depth)
        //    is visible. The "root depth" is the depth of the deepest node
        //    in the batch that shares the same prefix, i.e. batch.pos[0].
        //    Simpler rule: visible if KV pos <= q_pos AND has_seq_id(q_seq).
        for (int32_t i = 0; i < n_kv; ++i) {
            const auto & cell = kv_self.cells[i];
            if (cell.pos > q_pos)          continue;  // future
            if (!cell.has_seq_id(q_seq))   continue;  // wrong sequence
            data[q * stride_row + i] = 0.0f;
        }

        // 2) visibility among *batch* tokens (sibling mask): a tree node may
        //    attend to its ancestors in the batch but not its siblings.
        //    This only matters if the ancestors are ALSO in the batch (which
        //    is the case for depth > 1 tree nodes). The KV cells for those
        //    ancestors will have been written to the cache AT the position
        //    batch.pos[ancestor], with seq_id = batch.seq_id[ancestor][0].
        //    The rule "has_seq_id(q_seq)" already filters siblings correctly
        //    *provided* we assign distinct seq_ids to distinct tree branches
        //    that diverge. See §5 for the seq_id assignment.
    }
}
```

Explanation of the "simpler rule". Every tree node gets a unique `seq_id` per
branch path. The KV cells of the node's batch-level ancestors were written with
that same seq_id (because we `llama_kv_cache_seq_cp` the prefix before the
verify pass). Therefore `has_seq_id(q_seq)` returns true only for KV entries on
this node's ancestor chain, and the single causal comparison `cell.pos <= q_pos`
finishes the job. No explicit ancestor walk is needed — the KV metadata does
the tree reasoning for us.

The remaining fill work (SWA mask, embedding pooling) is untouched; we skip
only the default causal loop when `lctx.medusa_tree != NULL`.

### 3.6 `src/llama.cpp` — expose Medusa logits via context

In `struct llama_context`, add:

```cpp
    // Medusa output buffers, only populated when the loaded model has heads.
    // Size: n_medusa_heads, each of length n_outputs * n_vocab floats.
    std::vector<std::vector<float>> medusa_logits;
    const struct llama_medusa_tree * medusa_tree = nullptr;
```

In `llama_decode_internal`, after the graph compute completes and the main
`result_output` tensor is copied into `ctx->logits`, loop over Medusa heads and
copy out each `result_medusa_k` tensor into its slot:

```cpp
if (model.n_medusa_heads > 0) {
    lctx.medusa_logits.resize(model.n_medusa_heads);
    for (int k = 0; k < model.n_medusa_heads; ++k) {
        char name[64];
        snprintf(name, sizeof(name), "result_medusa_%d", k);
        ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        GGML_ASSERT(t != nullptr);
        const size_t n = (size_t) n_outputs * n_vocab;
        lctx.medusa_logits[k].resize(n);
        ggml_backend_tensor_get(t, lctx.medusa_logits[k].data(),
                                0, n * sizeof(float));
    }
}
```

And the public accessor:

```cpp
float * llama_get_medusa_logits_ith(struct llama_context * ctx,
                                    int32_t k, int32_t i) {
    if (!ctx || k < 0 || k >= (int32_t) ctx->medusa_logits.size())
        return nullptr;
    if (i < 0 || (size_t)(i + 1) * ctx->model.hparams.n_vocab
                 > ctx->medusa_logits[k].size())
        return nullptr;
    return ctx->medusa_logits[k].data() + (size_t) i * ctx->model.hparams.n_vocab;
}
```

### 3.7 New file: `examples/medusa/medusa.cpp`

A new standalone driver analogous to `examples/speculative/speculative.cpp`.
Skeleton (full implementation will follow this spec):

```cpp
int main(int argc, char ** argv) {
    // 1. Parse args, load single Medusa-enabled BitNet GGUF.
    // 2. Build the fixed Medusa-64 tree topology (static tables).
    // 3. Prefill prompt:
    //       llama_batch_add(batch, prompt_tokens, pos=[0..], seq=[0]);
    //       llama_decode(ctx, batch);
    // 4. Loop until EOS or max_new_tokens:
    //    a. Sample tree draft from main logits + Medusa head logits.
    //    b. Build verify batch (see §5 below) with tree_seq_ids and tree pos.
    //    c. llama_kv_cache_seq_cp the prefix to each branch seq_id.
    //    d. llama_set_tree_attention(ctx, &tree);
    //    e. llama_decode(ctx, batch_verify);
    //    f. Walk tree, accept longest match, commit KV via seq_rm.
    //    g. llama_set_tree_attention(ctx, nullptr); // back to causal
    //    h. Advance past the accepted tokens, print them.
}
```

### 3.8 New file: `examples/medusa/medusa_tree.h`

Declares the static Medusa-64 topology constants:

```cpp
// The static Medusa-64 tree from the original paper. Depth 4, 64 total nodes.
// parent[i] gives the index of node i's parent (-1 for the root).
// depth[i]  gives its depth relative to the root.
// choice[i] tells us which rank-m candidate of the parent produced it,
//           so we can pick tokens from each head's top-M predictions.
extern const int32_t MEDUSA64_N_NODES;
extern const int32_t MEDUSA64_PARENT[];
extern const int32_t MEDUSA64_DEPTH[];
extern const int32_t MEDUSA64_CHOICE[];
```

Populated from the Medusa paper's "medusa_choices" table.

### 3.9 Build system — `3rdparty/llama.cpp/examples/CMakeLists.txt`

Add one line to include the new example directory. We may also need an entry in
the top-level bitnet.cpp CMakeLists so the binary ends up alongside the existing
`llama-cli`.

---

## 4. Conversion pipeline end-to-end

1. **Train heads.** After the cache pass finishes and
   `data/hidden.bin` + `data/lm_head.pt` exist, run:
   ```
   python train.py --cached_hidden_path data/hidden.bin \
                   --cached_lm_head_path data/lm_head.pt \
                   --max_steps 2000 --log_every 10
   ```
   Produces `checkpoints/medusa_heads_step2000.pt`.
2. **Convert BitNet HF → GGUF.** Use bitnet.cpp's `setup_env.py` or
   `3rdparty/llama.cpp/convert_hf_to_gguf.py` to produce
   `models/bitnet-b1.58-2B.gguf` from the HF checkpoint.
3. **Merge Medusa heads into GGUF.**
   ```
   python MedusaBitNet/tools/convert_medusa_heads.py \
       --backbone_gguf models/bitnet-b1.58-2B.gguf \
       --heads_ckpt    checkpoints/medusa_heads_step2000.pt \
       --out_gguf      models/bitnet-b1.58-2B-medusa.gguf
   ```
4. **Run.**
   ```
   ./build/bin/llama-medusa -m models/bitnet-b1.58-2B-medusa.gguf \
       -p "Prompt text" -n 128
   ```

---

## 5. Tree construction and verify-batch layout

The tree is built exactly once at startup from the Medusa-64 tables. At each
generation step we use the same tree topology but different token contents.

### 5.1 Sampling the tree tokens

For each step, after the previous verify (or the prefill):

1. Take the *k* Medusa head logits at the last accepted token. Call them
   `L_0 .. L_{k-1}`, each `[n_vocab]`.
2. Keep top-M candidates from each head. The Medusa-64 tree uses
   `M = [top-1, top-6, top-4, top-2]` for depths 0, 1, 2, 3 — so each head
   contributes a fixed number of candidate children at its depth.
3. Walk the static tree, assigning token IDs: node 0 is the previously accepted
   token; each non-root node `i` gets the `choice[i]`-ranked token from head
   `depth[i] - 1`'s top-M list.

### 5.2 Batch filling

Let the global "absolute" position of the accepted prefix tail be `P` (known
from the prior decode). For `i ∈ [0, N_NODES)`:

```
batch.token[i]    = tree token for node i
batch.pos[i]      = P + depth[i]
batch.n_seq_id[i] = 1
batch.seq_id[i][0]= branch_seq[i]    // see below
batch.logits[i]   = true              // we verify every node
```

`branch_seq[i]` is the seq_id of the deepest branching path that node `i`
belongs to. Two nodes get the same seq_id iff one is an ancestor of the other
(i.e. they're on the same root→leaf path). In the Medusa-64 tree there are
~16-20 leaves, so we use `n_seq_max = 32` to be safe.

Algorithm to assign branch seq_ids to all 64 nodes:

```
leaves = [i for i in 0..N if no node has parent[i]]   // deepest nodes
next_seq = 0
branch_seq = [-1] * N
for leaf in leaves:
    path = walk parents from leaf to root
    # Assign this path a fresh seq_id if any node on it is still unassigned
    fresh_needed = any(branch_seq[n] == -1 for n in path)
    if fresh_needed:
        for n in path:
            if branch_seq[n] == -1:
                branch_seq[n] = next_seq
        next_seq += 1
```

With this rule each distinct root→leaf path gets one seq_id, and shared
prefix nodes carry the seq_id of the "first" leaf to claim them. Before
the verify decode we run:

```
for s in 1 .. next_seq-1:
    llama_kv_cache_seq_cp(ctx, 0, s, -1, P);  // share prefix up to P
```

so every branch sees the backbone prefill KV. After verify, we `seq_rm` every
position > `P + accepted_depth` on every seq_id and then `seq_keep(0)`.

### 5.3 Mask semantics sanity check

A tree node `q` at depth `d_q` with `branch_seq[q] = s_q` should attend to:

* All KV cells at positions `0..P` — these exist on seq 0 (prefill) and were
  copied to every `s_q` via `seq_cp`.
* All KV cells at positions `P+1..P+d_q-1` that belong to one of `q`'s
  ancestor nodes. Each ancestor has `branch_seq[ancestor] = s_q` (because
  ancestors share the leaf's path) or is on a *parent* path that was also
  copied. In practice, the seq_cp of prefixes plus the fresh assignment of
  seq_ids per path makes the per-KV `has_seq_id(s_q)` check correct without
  any explicit ancestor walking. See §3.5 for the code that relies on this.

---

## 6. Acceptance algorithm

```
pos_to_idx = { (depth[i], branch_seq[i]) : i  for i in range(N) }

current = 0                      # start at the root of the tree
accepted_tokens = []
while True:
    logits_here = llama_get_logits_ith(ctx, current)
    argmax_tok  = argmax(logits_here)

    # Try each immediate child and see if its token matches argmax_tok.
    best = None
    for c in children(current):
        if batch.token[c] == argmax_tok:
            best = c; break

    if best is None:
        # Backbone disagrees with every speculated child.
        accepted_tokens.append(argmax_tok)   # we still accept the backbone's own next token
        break

    accepted_tokens.append(argmax_tok)
    current = best
```

Medusa's trick: even when no speculation is accepted, we still get one real
next-token from the backbone forward for free (the main logits at the root).
So the worst case is exactly as fast as vanilla greedy decoding.

After the loop, commit the KV:

```
kept_path = [ancestors of 'current' including current]
for s in range(n_seq_max):
    if s == branch_seq[current]:
        # trim this seq to the accepted depth
        llama_kv_cache_seq_rm(ctx, s, P + len(accepted_tokens), -1);
    else:
        llama_kv_cache_seq_rm(ctx, s, P, -1);   // throw away entire branch
llama_kv_cache_seq_keep(ctx, branch_seq[current]);
# optional rename: seq_cp to seq 0 so subsequent steps stay on the canonical sequence.
```

---

## 7. Test plan (staged)

Each stage is runnable independently; we don't move to the next until the prior
one is green.

### Stage T0 — Converter round-trip (Python only, no C++ changes)

* Make a small fake backbone GGUF by running `convert_hf_to_gguf.py` on the
  HF BitNet checkpoint.
* Run `tools/convert_medusa_heads.py` on a synthetic `.pt` with known weights.
* Re-open the output GGUF with `gguf.GGUFReader` and assert every
  `medusa.head.*.weight` tensor matches the source `.pt` numerically
  (allclose to 1e-3 in F16).
* **Pass criterion:** shapes match, values match, metadata present.

### Stage T1 — C++ loader only

* Apply §3.2 and §3.3 edits.
* Build bitnet.cpp with its existing `llama-cli` binary.
* Run `llama-cli -m models/bitnet-b1.58-2B-medusa.gguf -p "hello"` with a
  Medusa-enabled GGUF.
* **Pass criterion:** model loads, logs show `"loaded 4 Medusa heads, 1 layers
  each"`, inference proceeds normally. The main logits and tokens generated
  should be bit-exact vs. the Medusa-free GGUF (Medusa heads are loaded but
  not used yet).

### Stage T2 — Graph build (Medusa logits produced, not consumed)

* Apply §3.4 and §3.6 edits.
* Add a temporary CLI flag `--dump-medusa` to the existing `llama-cli` that
  prints `argmax(llama_get_medusa_logits_ith(ctx, k, 0))` for each head after
  the prefill.
* Cross-check those argmaxes against running the PyTorch training-time model
  on the same prompt with the same trained heads — they should match exactly
  (bf16 rounding tolerance).
* **Pass criterion:** head-0 argmax equals the main-output argmax (because the
  Medusa paper specifies head 0 IS the normal next-token head after training),
  and heads 1..k-1 produce plausible top-1 tokens.

### Stage T3 — Tree mask correctness (no speculation yet)

* Apply §3.5 and §3.1 edits.
* Write a micro-test: construct a 3-node tree with known parent/child structure,
  feed a batch of 3 tokens, call `llama_decode`, and dump the attention mask
  tensor back from the host buffer. Assert the mask has exactly the expected
  0.0f / -INF pattern.
* **Pass criterion:** tree mask matches the hand-computed ground truth
  byte-for-byte.

### Stage T4 — End-to-end Medusa decoding

* Apply §3.7 and §3.8 edits.
* Run `llama-medusa -m models/bitnet-b1.58-2B-medusa.gguf -p "..." -n 64
  --tree medusa-64`.
* Compare generated text vs. greedy decoding from vanilla `llama-cli` on the
  same prompt. They should be **token-identical** when using the same
  sampler settings — Medusa only changes *how* tokens are computed, not what
  is chosen.
* **Pass criterion:** identical output, measurable tok/s speedup vs. baseline.

### Stage T5 — Throughput measurement

* Benchmark prompt: 2000 tokens of `The Pile` (or any natural text).
* Warm run, then time N=5 generations of 256 tokens each.
* Metrics: tok/s (mean), acceptance rate per step (mean/median/histogram),
  tree-position heatmap (how often each tree slot is on the accepted path).
* **Baseline:** same model, same prompt, vanilla bitnet.cpp (no Medusa).
* **Target:** 1.8×-3× speedup on the Z8; acceptance rate ≥ 2.0 tokens/step
  average for a 2k-step trained head, ≥ 2.5 for a 20k-step trained head.

---

## 8. Risks and open questions

* **R1 — BitNet kernels and the KQ mask.** BitNet's custom ops replace
  `GGML_OP_MUL_MAT` only on packed-ternary weights; attention (Q·Kᵀ softmax) is
  still standard ggml and consumes the F32 `inp_KQ_mask` through
  `ggml_soft_max_ext`. **Mitigation:** verified by reading the custom-op
  registration in `src/ggml-bitnet-lut.cpp`; no BitNet replacement for softmax.
  Tree mask should flow through unchanged. Confirm in Stage T3.

* **R2 — Flash attention cast.** `build_inp_KQ_mask` returns
  `ggml_cast(inp_KQ_mask, F16)` when `flash_attn` is enabled (line 10535). If
  we enable FA, the mask is rounded to F16 before attention — `-INFINITY`
  survives, but we should disable FA during Medusa for the initial
  implementation to reduce moving parts. Add a guard at startup:
  `if (model.n_medusa_heads > 0 && cparams.flash_attn) { LLAMA_LOG_WARN(...); cparams.flash_attn = false; }`.

* **R3 — Logit precision mismatch.** The Python training loop trains heads
  against bf16 hidden states with bf16 backbone weights and takes softmax in
  F32. bitnet.cpp runs heads in F16 (or F32 depending on the tensor dtype we
  store). The acceptance test at Stage T2 will catch any mismatch larger than
  a few bits.

* **R4 — KV cache size.** Medusa-64 expands one decode step into ≤ 64 KV
  writes across ~16-20 sequences. `llama_context` must be initialised with
  `n_seq_max = 32` and `n_ctx` large enough to accommodate the extra slots.
  Document in the example driver.

* **R5 — Sampling diversity.** The spec above assumes *greedy* accept/reject.
  Supporting temperature > 0 requires a probabilistic acceptance rule (e.g.
  the one from Leviathan et al.). Out of scope for the first version; revisit
  after greedy speedup is demonstrated.

* **R6 — Numerical fidelity of head conversion.** `torch.float16.tofile()` has
  worked in our favour so far, but we should add a `--dtype f32` fallback to
  the converter for early development, then switch to F16 for the production
  benchmark.

* **Open question Q1.** BitNet's `build_bitnet_158` calls
  `llm_build_lora_mm(lctx, ctx0, model.tok_embd, cur)` for the final vocab
  projection. If the model was loaded without a separate `output` tensor (the
  "tied embedding" branch at line 8728), `tok_embd` is the same tensor as
  `output`. We reuse `model.tok_embd` for Medusa too, which is correct in the
  tied case. If a future BitNet release ships with a separate `output`, we'll
  need to switch Medusa to use `model.output` instead.

* **Open question Q2.** Does bitnet.cpp's fork of llama.cpp already default
  `batch.n_seqs` > 1 for the decode path, or do we need to change the ubatch
  splitting logic to accept our verify batches? I couldn't confirm this from
  the headers alone; Stage T3 will exercise this in practice.

---

## 9. Work schedule

| Phase | Work                                                        | ETA       |
| ----- | ----------------------------------------------------------- | --------- |
| P0    | Cache pass running (background)                             | ~11h      |
| P1    | Train heads on cached features (2k-step smoke test)         | ~20 min   |
| P2    | Stage T0 — converter round-trip test                        | 1 hour    |
| P3    | §3.2 + §3.3 + Stage T1 (C++ loader)                         | 3-4 hours |
| P4    | §3.4 + §3.6 + Stage T2 (graph build)                        | 4-6 hours |
| P5    | §3.1 + §3.5 + Stage T3 (tree mask)                          | 4-6 hours |
| P6    | §3.7 + §3.8 + Stage T4 (end-to-end)                         | 4-8 hours |
| P7    | Stage T5 — benchmark vs. vanilla bitnet.cpp                 | 2 hours   |
| P8    | Train 20k-step heads, rebenchmark                           | ~1 hour   |

Realistic total (excluding the fixed 11h cache): 2-3 focused engineering days.
The risks in §8 dominate that estimate — any one of them could add another
day.

---

## 10. Out of scope (for now)

* GPU or NPU backends. The point is to prove the claim on CPU first.
* Distillation-based head training (Medusa-2). The Medusa-1 schema we're using
  (CE on future tokens against the dataset) is sufficient for a first result.
* Dynamic tree topology. We use the Medusa-64 static tree; the dynamic variant
  is a follow-up.
* Beam-searchy sampling. Greedy and top-k only.
