#!/usr/bin/env python3
"""Reproduces every experiment of the report.

    python3 scripts/run_experiments.py            # full study (about 1 hour on an Apple M4)
    python3 scripts/run_experiments.py --quick    # reduced sizes, a few minutes
    python3 scripts/run_experiments.py --only e1_scale,e6_delete --force

Steps:
  1. generate seeded synthetic datasets with build/bin/hnsw_gen (data/, not committed)
  2. run build/bin/hnsw_bench for each experiment group; each group writes
     results/<mode>/raw/<group>.csv (written to .partial and renamed when complete, so an
     interrupted run resumes with the unfinished groups only)
  3. record every executed command and the machine/compiler description

Plots and tables are produced separately by scripts/plot_results.py.
Optional real data: if data/sift/sift_base.fvecs exists (see scripts/download_sift.sh), the
SIFT1M groups run as well.
"""
import argparse
import os
import platform
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "bin")
DATA = os.path.join(ROOT, "data")


def sh(cmd, log):
    """Runs a command, mirrors it to the command log, aborts on failure."""
    line = " ".join(shlex.quote(c) for c in cmd)
    log.write(line + "\n")
    log.flush()
    print("$ " + line, flush=True)
    t = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"  ({time.time() - t:.1f} s)", flush=True)


class Plan:
    def __init__(self, quick):
        self.quick = quick
        # Scale factors: the quick mode keeps every experiment but shrinks the data 5-10x.
        self.n_small = 2_000 if quick else 10_000
        self.n_mid = 20_000 if quick else 100_000
        self.n_large = 100_000 if quick else 1_000_000
        self.nq = 200 if quick else 1000
        self.build_reps_small = 3 if quick else 5
        self.build_reps_mid = 2 if quick else 3
        self.query_reps = 3 if quick else 5

    def ds(self, kind, n, dim):
        return os.path.join(DATA, f"{kind}_d{dim}_n{n}")


def generate(plan, kind, n, dim, log, metric="l2"):
    prefix = plan.ds(kind, n, dim)
    if all(os.path.exists(prefix + suffix) for suffix in ("_base.fvecs", "_query.fvecs", "_gt.ivecs")):
        return prefix
    os.makedirs(DATA, exist_ok=True)
    cmd = [os.path.join(BIN, "hnsw_gen"), "--out", prefix, "--type", kind, "--n", str(n), "--nq", str(plan.nq),
           "--dim", str(dim), "--seed", "1", "--gt-k", "100", "--metric", metric]
    if kind == "clustered":
        cmd += ["--clusters", "64", "--latent", "16", "--spread", "0.1", "--noise", "0.01"]
    sh(cmd, log)
    return prefix


def sweep(prefix, out, log, tag, extra, gt=True):
    cmd = [os.path.join(BIN, "hnsw_bench"), "sweep", "--base", prefix + "_base.fvecs", "--query",
           prefix + "_query.fvecs", "--dataset", os.path.basename(prefix), "--out", out, "--tag", tag]
    if gt:
        cmd += ["--gt", prefix + "_gt.ivecs"]
    sh(cmd + extra, log)


EFS = "10,16,24,32,48,64,96,128,192,256,384,512"


def group_e1_scale(plan, out, log):
    """Scale: HNSW vs single-layer NSW vs exact brute force at three sizes."""
    for n, reps in ((plan.n_small, plan.build_reps_small), (plan.n_mid, plan.build_reps_mid), (plan.n_large, 1)):
        p = generate(plan, "clustered", n, 128, log)
        for index in ("hnsw", "nsw"):
            sweep(p, out, log, "e1", ["--index", index, "--efs", EFS, "--build-reps", str(reps),
                                       "--query-reps", str(plan.query_reps)])
        bf_reps = plan.query_reps if n < plan.n_large else 3
        sweep(p, out, log, "e1", ["--index", "bruteforce", "--query-reps", str(bf_reps)])


def group_e2_params(plan, out, log):
    """Graph degree M x construction beam efc: recall/latency/memory/build-time trade-off."""
    p = generate(plan, "clustered", plan.n_mid, 128, log)
    for m in (4, 8, 16, 32, 48):
        for efc in (100, 200, 400):
            sweep(p, out, log, "e2", ["--M", str(m), "--efc", str(efc), "--efs", EFS,
                                       "--build-reps", str(plan.build_reps_mid), "--query-reps", str(plan.query_reps)])


def group_e3_dim(plan, out, log):
    """Dimensionality: d = 32, 128, 960 at the middle scale."""
    for dim in (32, 128, 960):
        p = generate(plan, "clustered", plan.n_mid, dim, log)
        sweep(p, out, log, "e3", ["--efs", EFS, "--build-reps", str(plan.build_reps_mid),
                                   "--query-reps", str(plan.query_reps)])
        sweep(p, out, log, "e3", ["--index", "bruteforce", "--query-reps", str(plan.query_reps)])


def group_e4_k(plan, out, log):
    """Result size k = 1, 10, 100 on one index."""
    p = generate(plan, "clustered", plan.n_mid, 128, log)
    sweep(p, out, log, "e4", ["--ks", "1,10,100", "--efs", EFS + ",768,1024", "--query-reps", str(plan.query_reps)])


def group_e5_ablation(plan, out, log):
    """Neighbour selection (heuristic/simple) x hierarchy (HNSW/NSW) x data (clustered/uniform)."""
    for kind in ("clustered", "uniform"):
        p = generate(plan, kind, plan.n_mid, 128, log)
        for index in ("hnsw", "nsw"):
            for select in ("heuristic", "simple"):
                sweep(p, out, log, "e5", ["--index", index, "--select", select, "--efs", EFS + ",768,1024",
                                           "--query-reps", str(plan.query_reps)])


def group_e6_delete(plan, out, log):
    """Deleted fraction 0-50 % under tombstone / repair / rebuild: clustered data in random and
    hubs-first order, plus the hard uniform data in random order."""
    runs = [("clustered", "random"), ("clustered", "hubs"), ("uniform", "random")]
    for kind, order in runs:
        p = generate(plan, kind, plan.n_mid, 128, log)
        for policy in ("tombstone", "repair-prune", "repair-fixed", "repair", "rebuild"):
            cmd = [os.path.join(BIN, "hnsw_bench"), "delete", "--base", p + "_base.fvecs", "--query",
                   p + "_query.fvecs", "--dataset", os.path.basename(p), "--policy", policy, "--order", order,
                   "--fractions", "0,0.1,0.2,0.3,0.4,0.5", "--efs", "16,32,64,128", "--k", "10",
                   "--query-reps", str(plan.query_reps), "--out", out, "--tag", "e6"]
            sh(cmd, log)


def group_e7_churn(plan, out, log):
    """Steady replace workload: update rate x delete policy."""
    p = generate(plan, "clustered", plan.n_mid, 128, log)
    for rate in ("0.02", "0.10"):
        for policy in ("tombstone", "repair-prune", "repair", "rebuild"):
            cmd = [os.path.join(BIN, "hnsw_bench"), "churn", "--base", p + "_base.fvecs", "--query",
                   p + "_query.fvecs", "--dataset", os.path.basename(p), "--policy", policy, "--rate", rate,
                   "--initial", "0.5", "--rounds", "10", "--rebuild-every", "5", "--ef", "24", "--k", "10",
                   "--query-reps", "3", "--out", out, "--tag", "e7"]
            sh(cmd, log)


def group_e8_metric(plan, out, log):
    """Metric policy: L2 vs cosine vs inner product on the same vectors (exact GT per metric)."""
    p = generate(plan, "clustered", plan.n_mid, 128, log)
    for metric in ("l2", "cosine", "ip"):
        sweep(p, out, log, "e8", ["--metric", metric, "--efs", EFS + ",768,1024",
                                   "--query-reps", str(plan.query_reps)], gt=(metric == "l2"))


def group_e9_micro(plan, out_kernel, out_topk, log):
    """Microbenchmarks: SIMD vs scalar distance; quickselect vs heap vs full sort top-k."""
    sh([os.path.join(BIN, "hnsw_bench"), "kernel", "--dims", "32,128,960", "--reps", "5", "--out", out_kernel], log)
    ns = "10000,100000,1000000" if not plan.quick else "10000,100000"
    sh([os.path.join(BIN, "hnsw_bench"), "topk", "--ns", ns, "--ks", "1,10,100", "--reps", "5", "--out", out_topk], log)


def group_sift(plan, out, log):
    """Optional: SIFT1M (or its first 100K) with the published ground truth."""
    base = os.path.join(DATA, "sift", "sift_base.fvecs")
    if not os.path.exists(base):
        print("SIFT1M not found (run scripts/download_sift.sh); skipping", flush=True)
        return False
    q = os.path.join(DATA, "sift", "sift_query.fvecs")
    gt = os.path.join(DATA, "sift", "sift_groundtruth.ivecs")
    for n in (plan.n_mid, 1_000_000):
        cmd = [os.path.join(BIN, "hnsw_bench"), "sweep", "--base", base, "--query", q, "--n", str(n),
               "--dataset", f"sift_n{n}", "--efs", EFS, "--query-reps", str(plan.query_reps), "--out", out,
               "--tag", "sift"]
        if n == 1_000_000:
            cmd += ["--gt", gt]
        sh(cmd, log)
    return True


def group_sift_delete(plan, out, log):
    """Optional: deletion policies on the first 100K SIFT vectors (exact live-set ground truth).
    Uses data/sift/sift_base_100k.fvecs (scripts/download_sift.sh --base-100k) when present,
    otherwise the first rows of the full sift_base.fvecs."""
    base = os.path.join(DATA, "sift", "sift_base_100k.fvecs")
    if not os.path.exists(base):
        base = os.path.join(DATA, "sift", "sift_base.fvecs")
    if not os.path.exists(base) or not os.path.exists(os.path.join(DATA, "sift", "sift_query.fvecs")):
        print("SIFT data not found (scripts/download_sift.sh [--base-100k]); skipping", flush=True)
        return False
    q = os.path.join(DATA, "sift", "sift_query.fvecs")
    for policy in ("tombstone", "repair-prune", "repair-fixed", "repair", "rebuild"):
        sh([os.path.join(BIN, "hnsw_bench"), "delete", "--base", base, "--query", q, "--n", str(plan.n_mid),
            "--nq", str(plan.nq), "--dataset", f"sift_n{plan.n_mid}", "--policy", policy, "--order", "random",
            "--fractions", "0,0.1,0.2,0.3,0.4,0.5", "--efs", "16,32,64,128", "--k", "10",
            "--query-reps", str(plan.query_reps), "--out", out, "--tag", "sift_delete"], log)
    return True


GROUPS = ["e9_micro", "e1_scale", "e2_params", "e3_dim", "e4_k", "e5_ablation", "e6_delete", "e7_churn",
          "e8_metric", "sift", "sift_delete"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="reduced sizes (minutes instead of an hour)")
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(GROUPS))
    ap.add_argument("--force", action="store_true", help="rerun groups whose CSV already exists")
    args = ap.parse_args()
    for b in ("hnsw_gen", "hnsw_bench"):
        if not os.path.exists(os.path.join(BIN, b)):
            sys.exit(f"missing {b}; run `make` first")

    plan = Plan(args.quick)
    mode = "quick" if args.quick else "full"
    raw = os.path.join(ROOT, "results", mode, "raw")
    os.makedirs(raw, exist_ok=True)
    selected = [g for g in args.only.split(",") if g] or GROUPS
    with open(os.path.join(raw, "environment.txt"), "w") as env:
        env.write(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}\nplatform: {platform.platform()}\n")
        env.write(f"machine: {platform.machine()}\npython: {platform.python_version()}\n")
        try:
            cxx = subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        except (OSError, IndexError):
            cxx = "unknown"
        env.write(f"compiler: {cxx}\nflags: -std=c++17 -O3 (see Makefile), single-threaded queries\n")
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        except OSError:
            cpu = platform.processor()
        env.write(f"cpu: {cpu.strip()}\n")
    with open(os.path.join(raw, "commands.log"), "a") as log:
        log.write(f"# run started {time.strftime('%Y-%m-%d %H:%M:%S')} mode={mode}\n")
        for g in selected:
            if g not in GROUPS:
                sys.exit(f"unknown group {g}")
            if g == "e9_micro":
                outs = [os.path.join(raw, "e9_kernel.csv"), os.path.join(raw, "e9_topk.csv")]
                if all(os.path.exists(o) for o in outs) and not args.force:
                    print(f"skip {g} (exists)")
                    continue
                parts = [o + ".partial" for o in outs]
                for part in parts:
                    if os.path.exists(part):
                        os.remove(part)
                group_e9_micro(plan, parts[0], parts[1], log)
                for part, o in zip(parts, outs):
                    os.replace(part, o)
                continue
            out = os.path.join(raw, g + ".csv")
            if os.path.exists(out) and not args.force:
                print(f"skip {g} (exists)")
                continue
            part = out + ".partial"
            if os.path.exists(part):
                os.remove(part)
            t = time.time()
            print(f"=== {g}", flush=True)
            ran = globals()["group_" + g](plan, part, log)
            if os.path.exists(part):
                os.replace(part, out)
            elif ran is not False:
                print(f"warning: {g} produced no rows")
            print(f"=== {g} done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
