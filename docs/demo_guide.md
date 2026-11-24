# Live demo and code walkthrough guide (≈ 12 minutes)

## 0. Before the demo (once)

```bash
make -j8 && make test                       # show: 44 unit tests + 2 golden tests pass
./build/bin/hnsw_gen --out data/demo --type clustered --n 100000 --nq 1000 --dim 128
```

## 1. End-to-end workflow (Ruibo, 3 min)

```text
./build/bin/hnsw_cli
> build data/demo_base.fvecs --metric l2 --M 16 --efc 200      # ~8 s for 100K x 128
> info
> query data/demo_query.fvecs --k 10 --ef 10 --gt data/demo_gt.ivecs --show 3
> query data/demo_query.fvecs --k 10 --ef 50 --gt data/demo_gt.ivecs --show 3   # recall/latency trade-off
> set ef 50
> delete 0 1 2 3 4
> query data/demo_query.fvecs --k 10 --exact --show 0           # recall against the live set
> stats                                                          # tombstones, edges to tombstones, bytes/vector
> save data/demo.idx
> load data/demo.idx
> check
> compact
> build data/demo_base.fvecs --index bruteforce
> query data/demo_query.fvecs --k 10 --show 0                    # exact but ~100x slower
```

Error handling (index unchanged): `add 7 1,2,3`, `delete 999999999`, `load README.md`,
`query data/demo_query.fvecs --k ten`. Then `./build/bin/hnsw_cli --no-timing tests/golden/errors.cmds`.

## 2. Code walkthrough — who explains what

| Member | Walk through | Be ready to answer |
|---|---|---|
| Jing Ren | `HNSWIndex::add_prepared`, `insert_slot`, `select_neighbors`, `add_link`; `LevelGenerator::next` | Why P(level ≥ l) = M^-l? Why does the heuristic beat simple selection on clustered data (E5)? Why keep_pruned for inner product? How do M and efc trade build time/memory for recall (E2)? |
| Zhuoshen Cheng | `search`, `greedy_descend`, `search_layer` (MinHeap frontier, BoundedMaxHeap W, stop rule, tombstones traversed but not returned); `remove`, `repair_after_delete`, `choose_new_entry`; `distance.h` NEON/AVX2 kernels; `VisitedTable` | Why are tombstones still expanded? What does repair cost and why does it lose a little recall vs rebuild (E6)? What happens when the entry point is deleted? Why is distance a template policy and not a virtual call? |
| Ruibo Yang | `Index` interface + factory (virtual dispatch over templates), `BruteForceIndex` and top-k strategies, `OpenHashMap` (backward-shift delete), serializer (temp file + rename, checksum, bounds-checked load), CLI validation, test suite, `run_experiments.py` | How do you know HNSW is correct (brute force equality at ef ≥ N, invariant checker, interleaved test)? Why is the heap faster than quickselect (E9)? What does "index unchanged on error" mean in code? How were timings measured and repeated? |

## 3. Results slide (2 min)

Use `results/full/figures/e1_scale_recall_latency.png`, `e2_params_recall_latency.png`,
`e6_delete.png` and the numbers in `results/full/tables.md`.
