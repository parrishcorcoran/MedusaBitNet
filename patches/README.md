# Patches

`medusa-llama-cpp.patch` — full diff of the C++ edits that add Medusa
speculative decoding to bitnet.cpp's llama.cpp submodule. Touches:

- `include/llama.h`            — public Medusa API (4 functions + `llama_medusa_tree` struct)
- `src/llama.cpp`              — `llama_model` fields, loader for both
                                 `LLM_ARCH_BITNET` and `LLM_ARCH_BITNET_B158`,
                                 `build_bitnet` / `build_bitnet_158` graph
                                 tap point and Medusa residual blocks,
                                 `llama_decode_internal` Medusa logit extraction,
                                 and the four API implementations.
- `examples/CMakeLists.txt`    — registers the new `medusa` subdirectory.
- `examples/medusa/`           — the `llama-medusa` driver (linear chain topology).

Apply with:

```
cd bitnet.cpp/3rdparty/llama.cpp
git apply ../../../MedusaBitNet/patches/medusa-llama-cpp.patch
```

Regenerate after further edits with:

```
cd bitnet.cpp/3rdparty/llama.cpp
git add examples/CMakeLists.txt include/llama.h src/llama.cpp examples/medusa/
git diff --cached > ../../../MedusaBitNet/patches/medusa-llama-cpp.patch
git reset HEAD .
```
