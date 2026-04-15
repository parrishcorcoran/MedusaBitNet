"""Simulate wall-clock speedup for the full stack: ternary + skip gate + tree spec.

Clarifies which stack components contribute what, grounded in our empirical
measurements.

Cost model (CPU BitNet, measured):
  - Single-token forward: 1 unit
  - 5-token verify batch (Medusa): 3.3 units
  - 1-token forward + head eval: 1.08 units (F16 heads)
  - 1-token forward + head eval: 1.02 units (ternary heads)

Decoding modes simulated:
  A) Vanilla: 1 backbone forward per token → 1 token per 1 unit
  B) Medusa linear (current): 1 verify batch per step → (1 + accept_rate) per 3.3 units
  C) Medusa + trust-spec (skip verify when gate says): at gate hit rate g,
     emit all speculations without verify → higher tokens/unit
  D) Full stack: ternary heads + skip gate + tree spec

Uses our MEASURED acceptance (50.8%) and gate skip rates (4.35%-13.0%).
"""


def simulate(mode, params):
    """Return (tokens_per_unit_time, description)."""
    base_cost = params['base_cost']
    verify_cost = params['verify_cost']
    head_cost = params.get('head_cost', 0.08)
    accept_rate = params['accept_rate']  # head-k accepted avg
    gate_skip_rate = params.get('gate_skip_rate', 0.0)
    tree_factor = params.get('tree_factor', 1.0)  # multiplier on effective tokens
    fidelity = params.get('fidelity', 1.0)

    if mode == 'vanilla':
        return 1.0 / base_cost

    elif mode == 'medusa_linear':
        # Each step: cost verify_cost, produces 1 + accept_rate tokens
        tokens = 1 + accept_rate * tree_factor
        return tokens / verify_cost

    elif mode == 'medusa_skip_verify':
        # Gate hit (skip_rate fraction): no verify; emit 1 token at backbone-only cost
        # Gate miss: full verify batch
        g = gate_skip_rate
        cost_per_step = g * (base_cost + head_cost) + (1 - g) * verify_cost
        # When skip: emit 1 token. When not skip: 1 + accept_rate tokens
        tokens_per_step = g * 1 + (1 - g) * (1 + accept_rate * tree_factor)
        return tokens_per_step / cost_per_step

    elif mode == 'aggressive_spec':
        # Trust all K head speculations without verify when gate confident
        # K = 4 heads. When gate confident, emit K+1 tokens at base+head cost.
        # When not: emit 1 + accept_rate at verify cost.
        g = gate_skip_rate
        K = 4
        cost_per_step = g * (base_cost + head_cost) + (1 - g) * verify_cost
        tokens_per_step = g * (K + 1) + (1 - g) * (1 + accept_rate * tree_factor)
        return tokens_per_step / cost_per_step

    return None


def main():
    # Measured parameters (from our benchmarks)
    params = {
        'base_cost': 1.0,           # 1-token forward
        'verify_cost': 3.3,          # measured 5-token verify batch cost on CPU
        'head_cost': 0.08,           # F16 heads ~8% overhead
        'head_cost_ternary': 0.02,   # ternary heads ~2% overhead
        'accept_rate': 0.52,         # measured linear Medusa acceptance
    }

    print("=== BASELINE: vanilla ===")
    v = simulate('vanilla', params)
    print(f"  vanilla:         {v:.3f} tokens/unit\n")

    print("=== Linear Medusa (current) ===")
    m = simulate('medusa_linear', params)
    print(f"  F16 heads:       {m:.3f} tokens/unit  ({m/v:.2f}× vs vanilla)")

    p2 = dict(params)
    p2['head_cost'] = params['head_cost_ternary']
    m_tern = simulate('medusa_linear', p2)
    print(f"  Ternary heads:   {m_tern:.3f}  ({m_tern/v:.2f}× vs vanilla)  -- ternary alone")
    print()

    print("=== Medusa + skip-verify gate ===")
    print(f"(when gate confident, emit head-0 pred, skip verify batch)")
    print(f"{'skip_rate':>10} {'tok/unit':>10} {'vs_vanilla':>10}")
    for g in [0.05, 0.10, 0.20, 0.40]:
        p = dict(params); p['gate_skip_rate'] = g; p['head_cost'] = params['head_cost_ternary']
        s = simulate('medusa_skip_verify', p)
        print(f"  {g:>8.2%} {s:>10.3f} {s/v:>10.2f}×")
    print()

    print("=== Aggressive spec (trust all heads when gate confident) ===")
    print(f"(when gate confident, emit ALL K+1 tokens at backbone-only cost)")
    print(f"{'gate_rate':>10} {'tok/unit':>10} {'vs_vanilla':>10}  ; fidelity depends on gate τ")
    for g in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        p = dict(params); p['gate_skip_rate'] = g; p['head_cost'] = params['head_cost_ternary']
        a = simulate('aggressive_spec', p)
        print(f"  {g:>8.2%} {a:>10.3f} {a/v:>10.2f}×")
    print()

    print("=== Full stack: ternary + aggressive_spec + tree(×1.5 on accept) ===")
    print(f"(gate skip: trust all 4 heads; tree: higher acceptance when running)")
    for g in [0.10, 0.20, 0.40]:
        for tree in [1.0, 1.5, 2.0]:
            p = dict(params)
            p['head_cost'] = params['head_cost_ternary']
            p['gate_skip_rate'] = g
            p['tree_factor'] = tree
            a = simulate('aggressive_spec', p)
            print(f"  skip={g:.0%}  tree={tree:.1f}× → {a:.3f} tokens/unit ({a/v:.2f}× vanilla)")
    print()

    # What gate skip rate / tree combo gets us to 2× vanilla?
    print("=== Operating point for 2× vanilla (CPU moonshot step 1): ===")
    for target in [1.5, 2.0, 3.0, 5.0]:
        # Solve for needed skip rate given tree=1.5
        best = None
        for g in [x / 100 for x in range(0, 101)]:
            p = dict(params)
            p['head_cost'] = params['head_cost_ternary']
            p['gate_skip_rate'] = g
            p['tree_factor'] = 1.5
            ratio = simulate('aggressive_spec', p) / v
            if ratio >= target and best is None:
                best = g; break
        if best is not None:
            print(f"  {target}× vanilla needs gate skip rate ≥ {best:.2%} (with tree=1.5× acceptance)")
        else:
            print(f"  {target}× vanilla unreachable with tree=1.5× (need tree or accept boost)")


if __name__ == "__main__":
    main()
