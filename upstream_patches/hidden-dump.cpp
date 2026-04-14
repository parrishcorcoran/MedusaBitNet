// llama-hidden-dump: run a prompt, capture a specific named tensor per token
// (e.g. "l_out-30"), write raw float32 bytes to an output file.
//
// Use case: cache Medusa/Hydra training hidden states that actually match
// what the GGUF inference path sees at runtime.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct dump_ctx {
    std::string target_name;   // e.g. "l_out-30"
    FILE * out = nullptr;
    std::vector<uint8_t> scratch;
    size_t n_tensors_written = 0;
    size_t n_bytes_written = 0;
};

static bool cb_dump(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_ctx *) user_data;
    if (ask) {
        // only ask for the target tensor
        return d->target_name == t->name;
    }
    if (d->target_name != t->name) return true;

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    size_t n_bytes = ggml_nbytes(t);

    const void * src;
    if (is_host) {
        src = t->data;
    } else {
        d->scratch.resize(n_bytes);
        ggml_backend_tensor_get(t, d->scratch.data(), 0, n_bytes);
        src = d->scratch.data();
    }

    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "hidden-dump: refusing non-F32 tensor (type=%s)\n",
                ggml_type_name(t->type));
        return true;
    }

    fwrite(src, 1, n_bytes, d->out);
    d->n_tensors_written++;
    d->n_bytes_written += n_bytes;
    fprintf(stderr, "hidden-dump: wrote %s  shape=[%ld,%ld,%ld,%ld]  bytes=%zu\n",
            t->name, (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3], n_bytes);
    return true;
}

int main(int argc, char ** argv) {
    // Pull --layer and --dump-out out of argv before common_params_parse
    // so we don't confuse its flag parser.
    std::string tensor_name = "l_out-29";
    std::string dump_path = "hidden.bin";
    std::string tokens_path;  // if set, read pre-tokenized uint32 IDs from file
    std::vector<char *> filtered;
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) {
            tensor_name = argv[++i];
        } else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            tensor_name = std::string("l_out-") + argv[++i];
        } else if (strcmp(argv[i], "--dump-out") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens-file") == 0 && i + 1 < argc) {
            tokens_path = argv[++i];
        } else {
            filtered.push_back(argv[i]);
        }
    }
    int fargc = (int)filtered.size();
    char ** fargv = filtered.data();

    common_params params;
    if (!common_params_parse(fargc, fargv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    dump_ctx d;
    d.target_name = tensor_name;
    d.out = fopen(dump_path.c_str(), "wb");
    if (!d.out) {
        fprintf(stderr, "hidden-dump: failed to open %s\n", dump_path.c_str());
        return 1;
    }

    params.cb_eval = cb_dump;
    params.cb_eval_user_data = &d;
    params.warmup = false;

    common_init_result init = common_init_from_params(params);
    llama_model * model = init.model;
    llama_context * ctx = init.context;
    if (!model || !ctx) {
        fprintf(stderr, "hidden-dump: init failed\n");
        return 1;
    }

    std::vector<llama_token> tokens;
    if (!tokens_path.empty()) {
        FILE * tf = fopen(tokens_path.c_str(), "rb");
        if (!tf) { fprintf(stderr, "hidden-dump: cannot open %s\n", tokens_path.c_str()); return 1; }
        fseek(tf, 0, SEEK_END); long sz = ftell(tf); fseek(tf, 0, SEEK_SET);
        size_t n = sz / 4;
        std::vector<uint32_t> raw(n);
        fread(raw.data(), 4, n, tf); fclose(tf);
        tokens.reserve(n);
        for (auto id : raw) tokens.push_back((llama_token)id);
        fprintf(stderr, "hidden-dump: loaded %zu pre-tokenized IDs from %s\n", n, tokens_path.c_str());
    } else {
        const bool add_bos = llama_add_bos_token(model);
        tokens = common_tokenize(ctx, params.prompt, add_bos);
    }
    fprintf(stderr, "hidden-dump: prompt tokens = %zu, target = %s, out = %s\n",
            tokens.size(), d.target_name.c_str(), dump_path.c_str());

    // Build a batch with logits=true for every token so the graph actually
    // computes all per-token hidden states (not just the last).
    llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], (llama_pos)i, { 0 }, true);
    }
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "hidden-dump: decode failed\n");
        return 1;
    }
    llama_batch_free(batch);

    fclose(d.out);
    fprintf(stderr, "hidden-dump: done. tensors=%zu bytes=%zu\n",
            d.n_tensors_written, d.n_bytes_written);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
