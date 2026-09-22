# Source manifest

The core implementation was recovered from CNetDemo commit `6569b45262b488565a98901a8428980de91aa295`. Packaging imports, path handling, deterministic seeding, documentation, and the explicit `apply_graph_updates` compatibility switch were added during open-source cleanup.

SHA-256 of the original files at that commit:

```text
9d4ff79e72406a9c43887be5c8bf6bcc6ce43ff69531bf70f7b8d98df02049bb  my-flow/main.py
76a7a625c9671ca38ee14bd4b8ec136a1183b7afc027ba7ac6b498bbfd267ca7  my-flow/src/models/backbones/DualCongestionModel.py
8b8d9137a549208b681303dee4d57cc3fb437f74361a3820bded0389a33bccda  my-flow/src/models/heads/dual_congestion_head.py
bb786f12a35477ccdfd52a861c913645b3991adfea51ea9336aa3baa6c43682d  my-flow/src/solvers/base_solver.py
2e4084397fcebfdede1745cc36b8abf23efbcc8b8e0348e3e0262936ff5bc371  my-flow/src/solvers/finetune_solver.py
b47bdc8f268993715aeac2bbcafb3b0532c20ddd30c8c8108aec0dd8230b684d  my-flow/src/datasets/Dual_dataset.py
b069376e9d38e98b6a54dcc41c123603db0d839f5e604dc1a3d1c36adbd4cd46  my-flow/src/datasets/circuit_dataloader.py
5070d75ad2ae3d533bf80ce59e9291117d9988ff9978aecfdfec3c04786d3990  my-flow/scripts/Dual_data/extract_pixel_data.py
c4d0b424f0e7ebb025f64d714840e78db47953d2831dafde10ac46916403261b  my-flow/scripts/Dual_data/extract_net_data.py
```

The paper data uses the sparse-cell mode from the `b88f5f1` version of `extract_net_data.py`.
