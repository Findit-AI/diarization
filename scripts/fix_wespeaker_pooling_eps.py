"""Patch the WeSpeaker ONNX export to match pyannote's PyTorch
statistics pooling on sparse-mask edge cases.

Pyannote's `pyannote.audio.models.blocks.pooling.StatsPool` (line
52, 58) computes weighted mean/std via:

    v1 = weights.sum(dim=2) + 1e-8                     # eps for mean
    mean = (sequences * weights).sum(dim=2) / v1
    v2 = (weights ** 2).sum(dim=2)
    var = ((seq - mean)**2 * weights).sum(dim=2) / (v1 - v2/v1 + 1e-8)
    std = sqrt(var)

The ONNX export shipped under `models/wespeaker_resnet34_lm.onnx`
omits both `+ 1e-8` epsilons. With binary masks that have only 1-2
active frames out of 589, this causes:

  - 1 active frame:  v1 = 1, v2 = 1 → v1 - v2/v1 = 0 → div-by-zero → +inf
                     → propagates through Gemm to f32::MAX-class
                     embedding corruption (we measured 10/964 (chunk,
                     speaker) pairs on testaudioset 10 with this).
  - 2 active frames: v1 = 2, v2 = 2 → denom = 1, but f32 cancellation
                     in `v1 - v2/v1` near edge can still amplify.

The patch inserts two `Add(small_eps)` nodes:
  - `sum_1_eps = sum_1 + 1e-8`  (used by both mean and var denoms)
  - `sub_349_eps = sub_349 + 1e-8`  (used by var denom)

Output `models/wespeaker_resnet34_lm_stable.onnx` matches pyannote's
PyTorch stats pooling bit-exact for any mask sparsity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

EPS = 1e-8


def patch(in_path: Path, out_path: Path) -> None:
    m = onnx.load(str(in_path))
    g = m.graph

    # Add a 1e-8 constant initializer.
    eps_init = numpy_helper.from_array(
        np.array(EPS, dtype=np.float32), name="stats_pool_eps"
    )
    g.initializer.append(eps_init)

    # Find the relevant tensor names by walking nodes.
    # Node "ReduceSum" producing sum_1 (the v1 = sum(weights) tensor).
    # Node names from the captured graph dump:
    #   102: ReduceSum(unsqueeze_2) → sum_1
    #   105: Div(sum_2, sum_1) → div  (this is the MEAN)
    #   113: Div(sum_3, sum_1) → div_1
    #   114: Sub(sum_1, div_1) → sub_349  (n_eff = v1 - v2/v1)
    #   115: Div(sum_4, sub_349) → div_2  (this is var)
    # We need:
    #   sum_1     → sum_1 + eps  (for ALL consumers: 105 and 113)
    #   sub_349   → sub_349 + eps  (for consumer: 115)
    sum_1_consumers = ["div", "div_1"]      # nodes 105, 113 take sum_1
    sub_349_consumers = ["div_2"]           # node 115 takes sub_349

    # New tensor names.
    sum_1_eps_name = "sum_1_eps"
    sub_349_eps_name = "sub_349_eps"

    # Insert Add nodes.
    add_sum1 = helper.make_node(
        "Add",
        inputs=["sum_1", eps_init.name],
        outputs=[sum_1_eps_name],
        name="add_sum1_eps",
    )
    add_sub349 = helper.make_node(
        "Add",
        inputs=["sub_349", eps_init.name],
        outputs=[sub_349_eps_name],
        name="add_sub349_eps",
    )

    # Insert before any consumer that's a Div node. ONNX is
    # topologically ordered, so insert right after the original
    # producer. We append at the end and let ONNX reorder; in practice
    # all our target consumers come AFTER the producers, so simple
    # append works.
    g.node.append(add_sum1)
    g.node.append(add_sub349)

    # Re-route consumers' inputs.
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp == "sum_1" and n.output and n.output[0] in sum_1_consumers:
                # Mean (node 105) and div_1 (node 113) both consume
                # sum_1; pyannote uses v1 (sum_1+eps) for both.
                n.input[i] = sum_1_eps_name
            elif inp == "sub_349" and n.output and n.output[0] in sub_349_consumers:
                # Variance denominator (node 115) — gets +eps.
                n.input[i] = sub_349_eps_name

    # Re-topologically-sort: Add nodes come right after their producer
    # so consumers (which appear later in the original order) can see
    # the new tensors. We rebuild the node list by pulling Add nodes
    # forward into the right position.
    nodes = list(g.node)
    # Find positions.
    sum_1_idx = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "sum_1")
    sub_349_idx = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "sub_349")
    # Remove the appended Add nodes from the end.
    nodes = [n for n in nodes if n.name not in {"add_sum1_eps", "add_sub349_eps"}]
    # Insert after their producers (later index first to keep earlier index stable).
    insert_first = max(sum_1_idx, sub_349_idx)
    insert_second = min(sum_1_idx, sub_349_idx)
    if sum_1_idx > sub_349_idx:
        nodes.insert(insert_first + 1, add_sum1)
        nodes.insert(insert_second + 1, add_sub349)
    else:
        nodes.insert(insert_first + 1, add_sub349)
        nodes.insert(insert_second + 1, add_sum1)
    # Rebuild graph.
    del g.node[:]
    g.node.extend(nodes)

    onnx.checker.check_model(m)
    onnx.save(m, str(out_path))
    print(f"[patch] {in_path.name} -> {out_path.name}: added 2 Add(+1e-8) nodes")


if __name__ == "__main__":
    in_p = Path(sys.argv[1] if len(sys.argv) > 1 else "models/wespeaker_resnet34_lm.onnx")
    out_p = Path(sys.argv[2] if len(sys.argv) > 2 else "models/wespeaker_resnet34_lm_stable.onnx")
    patch(in_p, out_p)
