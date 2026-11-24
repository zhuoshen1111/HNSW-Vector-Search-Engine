#!/usr/bin/env python3
"""Turns results/<mode>/raw/*.csv into figures (results/<mode>/figures/*.png) and a Markdown
summary (results/<mode>/tables.md).

    python3 scripts/plot_results.py            # full results
    python3 scripts/plot_results.py --mode quick

Needs matplotlib (plots only; all measurements come from the C++ benchmark driver).
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference categorical palette (fixed order, light mode) and recessive ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.8, "lines.markersize": 4.5,
})


def load(raw, name):
    path = os.path.join(raw, name + ".csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def f(r, k):
    return float(r[k])


def curve(rows, x="recall", y="lat_mean_us"):
    pts = sorted(((f(r, x), f(r, y), r) for r in rows), key=lambda t: f(t[2], "ef"))
    return [p[0] for p in pts], [p[1] for p in pts]


def save(fig, out, name):
    fig.tight_layout()
    path = os.path.join(out, name + ".png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def label_end(ax, xs, ys, text, color):
    """Direct label at the last point of a line, in secondary ink with a colour swatch line."""
    if xs:
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                    color=INK2, va="center")


def min_latency_at(rows, target):
    """Smallest mean latency among sweep points whose recall >= target (None if never reached)."""
    ok = [f(r, "lat_mean_us") for r in rows if f(r, "recall") >= target]
    return min(ok) if ok else None


def fmt_us(v):
    return "—" if v is None else f"{v:.1f}"


# --------------------------------------------------------------------------------------------
def fig_e1(rows, out, md):
    if not rows:
        return
    ns = sorted({int(r["n"]) for r in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(3.3 * len(ns), 3.1), sharey=True)
    axes = axes if len(ns) > 1 else [axes]
    md.append("## E1 Scale: HNSW vs NSW vs brute force (clustered, d=128, M=16, efc=200, k=10)\n")
    md.append("| N | index | build s (median ± sd, reps) | bytes/vector | peak RSS MiB | min µs @ recall≥0.95 | min µs @ ≥0.99 | brute-force µs | speed-up @0.99 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for ax, n in zip(axes, ns):
        sub = [r for r in rows if int(r["n"]) == n]
        bf = [r for r in sub if r["index"] == "bruteforce"]
        bf_lat = f(bf[0], "lat_mean_us") if bf else None
        for i, idx in enumerate(("hnsw", "nsw")):
            s = [r for r in sub if r["index"] == idx]
            if not s:
                continue
            xs, ys = curve(s)
            ax.plot(xs, ys, marker="o", color=SERIES[i], label=idx.upper())
            r0 = s[0]
            m99 = min_latency_at(s, 0.99)
            md.append(f"| {n:,} | {idx} | {f(r0,'build_s'):.2f} ± {f(r0,'build_s_std'):.2f} ({r0['build_reps']}) | "
                      f"{f(r0,'bytes_per_vector'):.0f} | {int(r0['peak_rss_bytes'])/2**20:.0f} | "
                      f"{fmt_us(min_latency_at(s, 0.95))} | {fmt_us(m99)} | {fmt_us(bf_lat)} | "
                      f"{'—' if (m99 is None or bf_lat is None) else f'{bf_lat/m99:.0f}×'} |")
        if bf_lat:
            ax.axhline(bf_lat, color=INK2, lw=1.0, ls="--")
            ax.annotate("brute force (exact)", (0.0, bf_lat), xycoords=("axes fraction", "data"), xytext=(4, 3),
                        textcoords="offset points", fontsize=7.5, color=INK2)
            r0 = bf[0]
            md.append(f"| {n:,} | bruteforce | 0 | {f(r0,'bytes_per_vector'):.0f} | {int(r0['peak_rss_bytes'])/2**20:.0f} | "
                      f"{bf_lat:.1f} | {bf_lat:.1f} | {bf_lat:.1f} | 1× |")
        ax.set_yscale("log")
        ax.set_title(f"N = {n:,}")
        ax.set_xlabel("recall@10")
        ax.set_xlim(left=max(0.0, min(f(r, "recall") for r in sub if r["index"] != "bruteforce") - 0.02), right=1.005)
    axes[0].set_ylabel("mean query latency (µs, log)")
    axes[0].legend(loc="upper left")
    md.append("")
    save(fig, out, "e1_scale_recall_latency")

    # Build time and memory versus N.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.8))
    for i, idx in enumerate(("hnsw", "nsw")):
        s = sorted({int(r["n"]): r for r in rows if r["index"] == idx}.items())
        if not s:
            continue
        xs = [n for n, _ in s]
        a1.errorbar(xs, [f(r, "build_s") for _, r in s], yerr=[f(r, "build_s_std") for _, r in s], marker="o",
                    color=SERIES[i], label=idx.upper(), capsize=2)
        a2.plot(xs, [f(r, "bytes_per_vector") for _, r in s], marker="o", color=SERIES[i], label=idx.upper())
    bfm = sorted({int(r["n"]): r for r in rows if r["index"] == "bruteforce"}.items())
    if bfm:
        a2.plot([n for n, _ in bfm], [f(r, "bytes_per_vector") for _, r in bfm], marker="s", color=SERIES[2],
                label="brute force")
    a1.set_xscale("log"), a1.set_yscale("log"), a2.set_xscale("log")
    a1.set_xlabel("N (vectors)"), a1.set_ylabel("build time (s, log)"), a1.set_title("Build time")
    a2.set_xlabel("N (vectors)"), a2.set_ylabel("bytes per vector"), a2.set_title("Index memory (explicit estimate)")
    a1.legend(), a2.legend()
    save(fig, out, "e1_build_memory")


def fig_e2(rows, out, md):
    if not rows:
        return
    efcs = sorted({int(r["efc"]) for r in rows})
    ms = sorted({int(r["M"]) for r in rows})
    fig, axes = plt.subplots(1, len(efcs), figsize=(3.3 * len(efcs), 3.1), sharey=True)
    axes = axes if len(efcs) > 1 else [axes]
    for ax, efc in zip(axes, efcs):
        for i, m in enumerate(ms):
            s = [r for r in rows if int(r["M"]) == m and int(r["efc"]) == efc]
            xs, ys = curve(s)
            ax.plot(xs, ys, marker="o", color=SERIES[i], label=f"M={m}")
        ax.set_yscale("log"), ax.set_xlabel("recall@10"), ax.set_title(f"efc = {efc}")
        ax.set_xlim(0.80, 1.003)
    axes[0].set_ylabel("mean query latency (µs, log)")
    axes[0].legend(loc="upper left", ncol=1)
    save(fig, out, "e2_params_recall_latency")

    md.append("## E2 Graph degree M × construction beam efc (clustered, N=100K, d=128, k=10)\n")
    md.append("| M | efc | build s (median ± sd) | bytes/vector | avg layer-0 degree | min µs @0.95 | min µs @0.99 | min µs @0.999 | recall @ ef=10 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for m in ms:
        for efc in efcs:
            s = [r for r in rows if int(r["M"]) == m and int(r["efc"]) == efc]
            r0 = s[0]
            ef10 = [r for r in s if r["ef"] == "10"]
            md.append(f"| {m} | {efc} | {f(r0,'build_s'):.2f} ± {f(r0,'build_s_std'):.2f} | {f(r0,'bytes_per_vector'):.0f} | "
                      f"{f(r0,'avg_deg0'):.1f} | {fmt_us(min_latency_at(s,0.95))} | {fmt_us(min_latency_at(s,0.99))} | "
                      f"{fmt_us(min_latency_at(s,0.999))} | {f(ef10[0],'recall'):.3f} |" if ef10 else "")
    md.append("")

    # Build time and memory vs M.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.8))
    for i, efc in enumerate(efcs):
        s = sorted({int(r["M"]): r for r in rows if int(r["efc"]) == efc}.items())
        a1.errorbar([m for m, _ in s], [f(r, "build_s") for _, r in s], yerr=[f(r, "build_s_std") for _, r in s],
                    marker="o", color=SERIES[i], label=f"efc={efc}", capsize=2)
    s = sorted({int(r["M"]): r for r in rows if int(r["efc"]) == efcs[0]}.items())
    a2.plot([m for m, _ in s], [f(r, "bytes_per_vector") for _, r in s], marker="o", color=SERIES[0])
    a2.axhline(512, color=INK2, lw=1.0, ls="--")
    a2.annotate("raw vector (512 B)", (0.02, 512), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", fontsize=7.5, color=INK2)
    a1.set_xlabel("M"), a1.set_ylabel("build time (s)"), a1.set_title("Build time vs M"), a1.legend()
    a2.set_xlabel("M"), a2.set_ylabel("bytes per vector"), a2.set_title("Memory vs M")
    save(fig, out, "e2_params_build_memory")


def fig_e3(rows, out, md):
    if not rows:
        return
    dims = sorted({int(r["dim"]) for r in rows})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.9))
    md.append("## E3 Dimensionality (clustered, N=100K, M=16, efc=200, k=10)\n")
    md.append("| d | build s | bytes/vector | min µs @0.95 | min µs @0.99 | brute-force µs | speed-up @0.99 | dist. comps/query @0.99 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, d in enumerate(dims):
        s = [r for r in rows if int(r["dim"]) == d and r["index"] == "hnsw"]
        bf = [r for r in rows if int(r["dim"]) == d and r["index"] == "bruteforce"]
        xs, ys = curve(s)
        a1.plot(xs, ys, marker="o", color=SERIES[i], label=f"d={d}")
        xs2, ys2 = curve(s, y="dist_per_query")
        a2.plot(xs2, ys2, marker="o", color=SERIES[i], label=f"d={d}")
        bf_lat = f(bf[0], "lat_mean_us") if bf else None
        m99 = min_latency_at(s, 0.99)
        at99 = [r for r in s if f(r, "recall") >= 0.99]
        dc = min(f(r, "dist_per_query") for r in at99) if at99 else None
        md.append(f"| {d} | {f(s[0],'build_s'):.2f} | {f(s[0],'bytes_per_vector'):.0f} | {fmt_us(min_latency_at(s,0.95))} | "
                  f"{fmt_us(m99)} | {fmt_us(bf_lat)} | {'—' if not (m99 and bf_lat) else f'{bf_lat/m99:.0f}×'} | "
                  f"{'—' if dc is None else f'{dc:.0f}'} |")
    md.append("")
    a1.set_yscale("log"), a1.set_xlabel("recall@10"), a1.set_ylabel("mean latency (µs, log)")
    a1.set_title("Latency"), a1.legend(loc="upper left"), a1.set_xlim(0.8, 1.003)
    a2.set_yscale("log"), a2.set_xlabel("recall@10"), a2.set_ylabel("distance computations / query (log)")
    a2.set_title("Work (machine independent)"), a2.legend(loc="upper left"), a2.set_xlim(0.8, 1.003)
    save(fig, out, "e3_dimension")


def fig_e4(rows, out, md):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    md.append("## E4 Result size k (clustered, N=100K, d=128)\n")
    md.append("| k | recall@k at the smallest ef (effective ef) | effective ef for recall ≥ 0.99 | latency µs there |")
    md.append("|---|---|---|---|")
    for i, k in enumerate(sorted({int(r["k"]) for r in rows})):
        s = sorted([r for r in rows if int(r["k"]) == k], key=lambda r: int(r["ef"]))
        xs = [max(int(r["ef"]), k) for r in s]
        ys = [f(r, "recall") for r in s]
        ax.plot(xs, ys, marker="o", color=SERIES[i], label=f"k={k}")
        good = [r for r in s if f(r, "recall") >= 0.99]
        eff = f"{max(int(good[0]['ef']), k)}" if good else "—"
        lat = fmt_us(f(good[0], "lat_mean_us") if good else None)
        md.append(f"| {k} | {f(s[0], 'recall'):.3f} ({max(int(s[0]['ef']), k)}) | {eff} | {lat} |")
    md.append("")
    ax.set_xscale("log"), ax.set_xlabel("effective ef = max(ef, k) (log)"), ax.set_ylabel("recall@k")
    ax.set_title("Recall@k vs search breadth"), ax.legend(loc="lower right")
    save(fig, out, "e4_k")


def fig_e5(rows, out, md):
    if not rows:
        return
    kinds = [k for k in ("clustered", "uniform") if any(k in r["dataset"] for r in rows)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.1))
    axes = axes if len(kinds) > 1 else [axes]
    variants = [("hnsw", "heuristic"), ("hnsw", "simple"), ("nsw", "heuristic"), ("nsw", "simple")]
    md.append("## E5 Ablation: hierarchy × neighbour selection × data (N=100K, d=128, M=16, efc=200)\n")
    md.append("| data | index | selection | build s | recall @ef=64 | max recall (ef≤1024) | dist/query @ef=64 | min µs @0.95 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ax, kind in zip(axes, kinds):
        for i, (idx, sel) in enumerate(variants):
            s = [r for r in rows if kind in r["dataset"] and r["index"] == idx and r["select"] == sel]
            if not s:
                continue
            xs, ys = curve(s, y="dist_per_query")
            ax.plot(xs, ys, marker="o", color=SERIES[i], label=f"{idx.upper()} {sel}")
            e64 = [r for r in s if r["ef"] == "64"]
            md.append(f"| {kind} | {idx} | {sel} | {f(s[0],'build_s'):.2f} | {f(e64[0],'recall'):.3f} | "
                      f"{max(f(r,'recall') for r in s):.3f} | {f(e64[0],'dist_per_query'):.0f} | "
                      f"{fmt_us(min_latency_at(s,0.95))} |" if e64 else "")
        ax.set_yscale("log"), ax.set_xlabel("recall@10"), ax.set_title(f"{kind} data")
        ax.set_ylabel("distance computations / query (log)")
    axes[0].legend(loc="upper left")
    md.append("")
    save(fig, out, "e5_ablation")


POLICIES = ["tombstone", "repair-prune", "repair-fixed", "repair", "rebuild"]


def policy_color(pol):
    return SERIES[POLICIES.index(pol)]  # colour follows the policy in every figure


def fig_e6(rows, out, md):
    if not rows:
        return
    panels = [("clustered", "random", "16"), ("clustered", "hubs", "16"), ("uniform", "random", "64"),
              ("sift", "random", "16")]
    panels = [p for p in panels if any(p[0] in r["dataset"] and r["order"] == p[1] for r in rows)]
    nrow = 3
    fig, axes = plt.subplots(nrow, len(panels), figsize=(3.1 * len(panels), 7.6), sharex=True)
    if len(panels) == 1:
        axes = [[a] for a in axes]
    md.append("## E6 Deletion policies (N=100K, d=128, k=10; sift = first 100K SIFT vectors)\n")
    md.append("| data / order | policy | ef | recall @0% | @10% | @30% | @50% | µs/query @0% → @50% | dist/query @0% → @50% | "
              "avg out-degree @50% | unreachable @50% | delete µs/op (avg) | rebuild s @50% | edges→tombstones @50% |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for col, (kind, order, ef) in enumerate(panels):
        for pol in POLICIES:
            s = sorted([r for r in rows if kind in r["dataset"] and r["order"] == order and r["policy"] == pol
                        and r["ef"] == ef], key=lambda r: f(r, "fraction"))
            if not s:
                continue
            xs = [100 * f(r, "fraction") for r in s]
            c = policy_color(pol)
            axes[0][col].plot(xs, [f(r, "recall") for r in s], marker="o", color=c, label=pol)
            axes[1][col].plot(xs, [f(r, "lat_mean_us") for r in s], marker="o", color=c, label=pol)
            if "unreachable_live" in s[0]:
                axes[2][col].plot(xs, [f(r, "unreachable_live") for r in s], marker="o", color=c, label=pol)
            by = {round(f(r, "fraction"), 2): r for r in s}
            dels = [f(r, "delete_us_per_op") for r in s if f(r, "fraction") > 0]
            g = lambda fr, key: f"{f(by[fr], key):.3f}" if fr in by else "—"  # noqa: E731
            md.append(f"| {kind} / {order} | {pol} | {ef} | {g(0.0,'recall')} | {g(0.1,'recall')} | {g(0.3,'recall')} | "
                      f"{g(0.5,'recall')} | {f(s[0],'lat_mean_us'):.1f} → {f(s[-1],'lat_mean_us'):.1f} | "
                      f"{f(s[0],'dist_per_query'):.0f} → {f(s[-1],'dist_per_query'):.0f} | {f(s[-1],'avg_deg0'):.1f} | "
                      f"{s[-1].get('unreachable_live', '—')} | {sum(dels)/len(dels):.2f} | {f(s[-1],'rebuild_s'):.2f} | "
                      f"{s[-1]['edges_to_tombstones']} |")
        axes[0][col].set_title(f"{kind}, {order} order (ef={ef})")
        axes[0][col].set_ylabel("recall@10 (live set)")
        axes[1][col].set_ylabel("mean latency (µs)")
        axes[2][col].set_ylabel("unreachable live vectors")
        axes[2][col].set_xlabel("deleted fraction (%)")
    axes[0][0].legend(loc="lower left", fontsize=7)
    md.append("")
    save(fig, out, "e6_delete")


def fig_e7(rows, out, md):
    if not rows:
        return
    rates = sorted({r["rate"] for r in rows}, key=float)
    fig, axes = plt.subplots(2, len(rates), figsize=(3.6 * len(rates), 5.0), sharex=True)
    if len(rates) == 1:
        axes = [[axes[0]], [axes[1]]]
    md.append("## E7 Churn: steady replace workload (pool N=100K, 50K live, ef=24, k=10)\n")
    md.append("| update rate / round | policy | recall round 0 → 10 (min) | µs/query round 0 → 10 | update µs/op | tombstones after 10 rounds |")
    md.append("|---|---|---|---|---|---|")
    for col, rate in enumerate(rates):
        for pol in POLICIES:
            s = sorted([r for r in rows if r["rate"] == rate and r["policy"] == pol], key=lambda r: int(r["round"]))
            if not s:
                continue
            xs = [int(r["round"]) for r in s]
            axes[0][col].plot(xs, [f(r, "recall") for r in s], marker="o", color=policy_color(pol), label=pol)
            axes[1][col].plot(xs, [f(r, "lat_mean_us") for r in s], marker="o", color=policy_color(pol), label=pol)
            upd = [f(r, "update_us_per_op") for r in s[1:]]
            md.append(f"| {float(rate)*100:.0f}% | {pol} | {f(s[0],'recall'):.4f} → {f(s[-1],'recall'):.4f} "
                      f"({min(f(r,'recall') for r in s):.4f}) | {f(s[0],'lat_mean_us'):.1f} → {f(s[-1],'lat_mean_us'):.1f} | "
                      f"{sum(upd)/len(upd):.1f} | {s[-1]['tombstones']} |")
        axes[0][col].set_title(f"{float(rate)*100:.0f}% of live vectors replaced per round")
        axes[0][col].set_ylabel("recall@10")
        axes[1][col].set_ylabel("mean latency (µs)")
        axes[1][col].set_xlabel("round")
    axes[0][0].legend(loc="lower left")
    md.append("")
    save(fig, out, "e7_churn")


def fig_e8(rows, out, md):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    md.append("## E8 Metric policies on the same vectors (N=100K, d=128, M=16, efc=200)\n")
    md.append("| metric | recall @ef=50 | min µs @0.95 | min µs @0.99 | avg layer-0 degree |")
    md.append("|---|---|---|---|---|")
    for i, m in enumerate(("l2", "cosine", "ip")):
        s = [r for r in rows if r["metric"] == m]
        if not s:
            continue
        xs, ys = curve(s)
        ax.plot(xs, ys, marker="o", color=SERIES[i], label={"l2": "L2", "cosine": "cosine", "ip": "inner product"}[m])
        e48 = [r for r in s if r["ef"] == "48"] or s
        md.append(f"| {m} | {f(e48[0],'recall'):.3f} (ef={e48[0]['ef']}) | {fmt_us(min_latency_at(s,0.95))} | "
                  f"{fmt_us(min_latency_at(s,0.99))} | {f(s[0],'avg_deg0'):.1f} |")
    md.append("")
    ax.set_yscale("log"), ax.set_xlabel("recall@10"), ax.set_ylabel("mean latency (µs, log)")
    ax.set_title("Metric policy"), ax.legend(loc="upper left")
    save(fig, out, "e8_metric")


def fig_e9(kernel, topk, out, md):
    if kernel:
        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        dims = sorted({int(r["dim"]) for r in kernel})
        width = 0.38
        for i, impl in enumerate(("scalar", "simd")):
            vals = [f(next(r for r in kernel if int(r["dim"]) == d and r["kernel"] == "l2" and r["impl"] == impl),
                      "ns_per_distance") for d in dims]
            xs = [j + (i - 0.5) * (width + 0.02) for j in range(len(dims))]
            ax.bar(xs, vals, width=width, color=SERIES[i], label=impl if impl == "scalar" else
                   f"SIMD ({kernel[0]['simd']})")
            for x, v in zip(xs, vals):
                ax.annotate(f"{v:.0f}", (x, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7,
                            color=INK2)
        ax.set_xticks(range(len(dims)), [f"d={d}" for d in dims])
        ax.set_yscale("log"), ax.set_ylabel("ns per L2 distance (log)"), ax.set_title("Distance kernel")
        ax.legend(loc="upper left")
        save(fig, out, "e9_kernel")
        md.append("## E9a Distance kernels (ns per distance, median of 5)\n")
        md.append("| d | kernel | scalar ns | SIMD ns | speed-up | max relative difference |")
        md.append("|---|---|---|---|---|---|")
        for d in dims:
            for kname in ("l2", "dot"):
                sc = next(r for r in kernel if int(r["dim"]) == d and r["kernel"] == kname and r["impl"] == "scalar")
                si = next(r for r in kernel if int(r["dim"]) == d and r["kernel"] == kname and r["impl"] == "simd")
                md.append(f"| {d} | {kname} | {f(sc,'ns_per_distance'):.1f} | {f(si,'ns_per_distance'):.1f} | "
                          f"{f(sc,'ns_per_distance')/f(si,'ns_per_distance'):.1f}× | {f(si,'max_rel_error'):.1e} |")
        md.append("")
    if topk:
        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        for i, m in enumerate(("quickselect", "heap", "fullsort")):
            s = sorted([r for r in topk if r["method"] == m and r["k"] == "10"], key=lambda r: int(r["n"]))
            ax.plot([int(r["n"]) for r in s], [f(r, "median_us") for r in s], marker="o", color=SERIES[i], label=m)
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlabel("n candidates (log)")
        ax.set_ylabel("selection time (µs, log)"), ax.set_title("Exact top-10 selection"), ax.legend(loc="upper left")
        save(fig, out, "e9_topk")
        md.append("## E9b Exact top-k selection (median µs of 5; selection only)\n")
        md.append("| n | k | quickselect | bounded heap | full sort |")
        md.append("|---|---|---|---|---|")
        for n in sorted({int(r["n"]) for r in topk}):
            for k in sorted({int(r["k"]) for r in topk}):
                v = {r["method"]: f(r, "median_us") for r in topk if int(r["n"]) == n and int(r["k"]) == k}
                md.append(f"| {n:,} | {k} | {v['quickselect']:.0f} | {v['heap']:.0f} | {v['fullsort']:.0f} |")
        md.append("")


def fig_sift(rows, out, md):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    md.append("## SIFT (real data, M=16, efc=200, k=10)\n")
    md.append("| dataset | build s | bytes/vector | recall @ef=16 | min µs @0.95 | min µs @0.99 |")
    md.append("|---|---|---|---|---|---|")
    for i, ds in enumerate(sorted({r["dataset"] for r in rows})):
        s = [r for r in rows if r["dataset"] == ds]
        xs, ys = curve(s)
        ax.plot(xs, ys, marker="o", color=SERIES[i], label=ds)
        e16 = [r for r in s if r["ef"] == "16"]
        md.append(f"| {ds} | {f(s[0],'build_s'):.1f} | {f(s[0],'bytes_per_vector'):.0f} | "
                  f"{f(e16[0],'recall'):.3f} | {fmt_us(min_latency_at(s,0.95))} | {fmt_us(min_latency_at(s,0.99))} |" if e16 else "")
    md.append("")
    ax.set_yscale("log"), ax.set_xlabel("recall@10"), ax.set_ylabel("mean latency (µs, log)")
    ax.set_title("SIFT"), ax.legend(loc="upper left")
    save(fig, out, "sift")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "quick"])
    args = ap.parse_args()
    base = os.path.join(ROOT, "results", args.mode)
    raw = os.path.join(base, "raw")
    out = os.path.join(base, "figures")
    os.makedirs(out, exist_ok=True)
    md = [f"# Experiment tables ({args.mode} run)\n",
          "Generated by `scripts/plot_results.py` from `results/%s/raw/*.csv`. Latencies are single-thread "
          "wall-clock means per query (median over repetitions); memory is the explicit data-structure estimate "
          "unless labelled RSS.\n" % args.mode]
    env = os.path.join(raw, "environment.txt")
    if os.path.exists(env):
        md.append("```\n" + open(env).read().strip() + "\n```\n")
    fig_e1(load(raw, "e1_scale"), out, md)
    fig_e2(load(raw, "e2_params"), out, md)
    fig_e3(load(raw, "e3_dim"), out, md)
    fig_e4(load(raw, "e4_k"), out, md)
    fig_e5(load(raw, "e5_ablation"), out, md)
    fig_e6(load(raw, "e6_delete") + load(raw, "sift_delete"), out, md)
    fig_e7(load(raw, "e7_churn"), out, md)
    fig_e8(load(raw, "e8_metric"), out, md)
    fig_e9(load(raw, "e9_kernel"), load(raw, "e9_topk"), out, md)
    fig_sift(load(raw, "sift"), out, md)
    with open(os.path.join(base, "tables.md"), "w") as fh:
        fh.write("\n".join(line for line in md if line is not None) + "\n")
    print(f"wrote {out}/*.png and {base}/tables.md")


if __name__ == "__main__":
    main()
