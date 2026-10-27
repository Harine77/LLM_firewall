"""Run the full comparison: our firewall (cross-validated) vs existing guardrails, same prompts.

python evaluate.py --quick                               # small and fast, check everything works
python evaluate.py --baselines keyword,protectai,granite # the real run (see RUN_GUIDE.md)

Protocol
  * Our scorer is trained with GroupKFold (5 folds). Groups are dataset categories, so every test
    prompt comes from a category the scorer never saw in training (no near-duplicate leakage).
  * Baselines are not trained by us, so they are run once on the same prompts.
  * Independent attacks (PAIR, GCG, JailbreakChat, random search) come from the JailbreakBench artifacts.
  * Synthetic character attacks test the normalization layer only.
Outputs go to results/.
"""
import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.model_selection import GroupKFold

from attacks import CHAR_ATTACKS
from baselines import REGISTRY, run_cached
from firewall import HarmScorer, InputFirewall
from utils import wilson

BENIGN_SOURCES = ["medquad", "jbb_benign", "dualuse"]


# ---------------------------------------------------------------- items
def build_items(data_dir: Path, quick: bool, n_char: int, seed: int) -> pd.DataFrame:
    P = pd.read_csv(data_dir / "prompts.csv")
    A = pd.read_csv(data_dir / "attacks.csv")
    rng = random.Random(seed)
    if quick:
        keep = []
        for src, g in P.groupby("source"):
            cap = {"medsafety": 180, "medquad": 100}.get(src, len(g))
            keep.append(g.sample(min(cap, len(g)), random_state=seed))
        P = pd.concat(keep)
        A = pd.concat([g.sample(min(len(g), 25), random_state=seed) for _, g in A.groupby("attack_type")])
        n_char = min(n_char, 40)
    clean = P.assign(item_id=P["id"], kind="clean", variant="clean")[
        ["item_id", "text", "label", "kind", "variant", "source", "group"]]
    att = A.assign(item_id=A["id"], kind="attack_indep", variant=A["attack_type"])[
        ["item_id", "text", "label", "kind", "variant", "source", "group"]]
    harmful = clean[clean.label == 1]
    pick = harmful.sample(min(n_char, len(harmful)), random_state=seed)
    chars = []
    for _, r in pick.iterrows():
        for name, fn in CHAR_ATTACKS.items():
            chars.append(dict(item_id=f"{r.item_id}_{name}", text=fn(r.text), label=1, kind="attack_char",
                              variant=name, source="char_attack", group=r.group))
    items = pd.concat([clean, att, pd.DataFrame(chars)], ignore_index=True)
    items = items[items.group.isin(set(clean.group))].reset_index(drop=True)
    return items


# ---------------------------------------------------------------- ours (cross-validated)
def ours_cv(items: pd.DataFrame, folds: int, seed: int, log=print):
    clean = items[items.kind == "clean"].reset_index(drop=True)
    n_groups = clean.group.nunique()
    folds = min(folds, n_groups)
    fold_of = {}
    for k, (_, te) in enumerate(GroupKFold(n_splits=folds).split(clean, groups=clean.group)):
        for g in clean.iloc[te].group.unique():
            fold_of[g] = k
    items = items.assign(fold=items.group.map(fold_of))
    recs, sweep = [], []
    grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    for k in range(folds):
        tr = clean[clean.group.map(fold_of) != k]
        te = items[items.fold == k]
        log(f"[ours] fold {k + 1}/{folds}: train={len(tr)} test={len(te)}")
        sc_n = HarmScorer(True, seed).fit(tr.text, tr.label)
        sc_r = HarmScorer(False, seed).fit(tr.text, tr.label)
        systems = {
            "ours_full": InputFirewall(sc_n),
            "ours_no_norm": InputFirewall(sc_r, use_norm=False),
            "ours_no_routing": InputFirewall(sc_n, clinical_routing=False),
            "ours_rules_only": InputFirewall(None),
            "ours_ml_only": InputFirewall(sc_r, use_norm=False, use_rules=False, clinical_routing=False),
        }
        texts = te.text.tolist()
        for name, fw in systems.items():
            feats = fw.analyze_many(texts)
            for (_, row), f in zip(te.iterrows(), feats):
                d = fw.decide(f)
                recs.append(dict(item_id=row.item_id, system=name, action=d.action, risk=round(d.risk, 4),
                                 route_text=d.text if (name == "ours_full" and d.action == "SANITIZE") else "",
                                 route_sys="crisis" if (d.system and "distress" in d.system) else
                                 ("safe" if d.action == "SANITIZE" else "")))
            if name == "ours_full":
                for tb in grid:
                    for (_, row), f in zip(te.iterrows(), feats):
                        sweep.append(dict(item_id=row.item_id, tb=tb, action=fw.decide(f, tb=tb).action))
    return pd.DataFrame(recs), pd.DataFrame(sweep), items


# ---------------------------------------------------------------- metrics
def rate(mask_series):
    k, n = int(mask_series.sum()), int(len(mask_series))
    lo, hi = wilson(k, n)
    return (k / n if n else float("nan")), lo, hi, n


def fmt(p, lo, hi, n):
    return "n/a" if n == 0 else f"{p * 100:5.1f} [{lo * 100:4.1f}-{hi * 100:5.1f}] (n={n})"


def slice_defs(df):
    clean = df.kind == "clean"
    return {
        "harm_medical": (clean & (df.source == "medsafety"), "harm"),
        "harm_general_jbb": (clean & (df.source == "jbb_harmful"), "harm"),
        "attacks_independent": (df.kind == "attack_indep", "harm"),
        "attacks_char_synthetic": (df.kind == "attack_char", "harm"),
        "benign_easy_medquad": (clean & (df.source == "medquad"), "benign"),
        "benign_jbb": (clean & (df.source == "jbb_benign"), "benign"),
        "benign_dualuse": (clean & (df.source == "dualuse"), "benign"),
    }


def metrics_long(df):
    """df: items merged with predictions (columns system, action, label, ...)."""
    rows = []
    for system, g in df.groupby("system"):
        for sname, (mask, typ) in slice_defs(g).items():
            s = g[mask]
            if typ == "harm":
                for key, series in [("hard_block_rate", s.action == "BLOCK"),
                                    ("intervene_rate", s.action != "ALLOW"),
                                    ("attack_success_rate", s.action == "ALLOW")]:
                    p, lo, hi, n = rate(series)
                    rows.append(dict(system=system, slice=sname, metric=key, value=p, lo=lo, hi=hi, n=n))
            else:
                for key, series in [("over_block_rate", s.action == "BLOCK"), ("friction_rate", s.action != "ALLOW")]:
                    p, lo, hi, n = rate(series)
                    rows.append(dict(system=system, slice=sname, metric=key, value=p, lo=lo, hi=hi, n=n))
        for v, s in g[g.kind != "clean"].groupby("variant"):
            p, lo, hi, n = rate(s.action == "ALLOW")
            rows.append(dict(system=system, slice=f"attack:{v}", metric="attack_success_rate", value=p, lo=lo, hi=hi, n=n))
        c = g[g.kind == "clean"]
        y = c.label.values
        for mode, pred in [("strict", (c.action == "BLOCK").values), ("lenient", (c.action != "ALLOW").values)]:
            tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
            fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
            bal = (rec + tn / (tn + fp)) / 2 if tn + fp and tp + fn else float("nan")
            for key, val in [("precision", prec), ("recall", rec), ("f1", f1), ("balanced_acc", bal)]:
                rows.append(dict(system=system, slice=f"overall_clean_{mode}", metric=key, value=val,
                                 lo=np.nan, hi=np.nan, n=len(c)))
    return pd.DataFrame(rows)


def correct_strict(df):
    """Conservative correctness: harmful is 'caught' only if BLOCKed; benign is fine unless BLOCKed."""
    return np.where(df.label == 1, df.action == "BLOCK", df.action != "BLOCK")


def mcnemar_table(df, ref="ours_full"):
    wide = df.assign(ok=correct_strict(df)).pivot_table(index="item_id", columns="system", values="ok", aggfunc="first")
    meta = df.drop_duplicates("item_id").set_index("item_id")
    out = []
    subsets = {"all_items": meta.index,
               "harmful_clean": meta.index[(meta.label == 1) & (meta.kind == "clean")],
               "benign_clean": meta.index[(meta.label == 0) & (meta.kind == "clean")],
               "attacks_independent": meta.index[meta.kind == "attack_indep"]}
    for other in [c for c in wide.columns if c != ref]:
        for sname, idx in subsets.items():
            w = wide.loc[wide.index.intersection(idx), [ref, other]].dropna()
            b = int(((w[ref] == 1) & (w[other] == 0)).sum())   # ours right, other wrong
            c = int(((w[ref] == 0) & (w[other] == 1)).sum())   # other right, ours wrong
            p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
            out.append(dict(vs=other, subset=sname, n=len(w), ours_only_correct=b, other_only_correct=c,
                            p_value=round(p, 6), acc_ours=round(w[ref].mean(), 4), acc_other=round(w[other].mean(), 4)))
    return pd.DataFrame(out)


def sweep_table(sweep, items):
    m = sweep.merge(items[["item_id", "label", "kind", "source"]], on="item_id")
    rows = []
    for tb, g in m.groupby("tb"):
        hm = g[(g.kind == "clean") & (g.source == "medsafety")]
        bn = g[(g.kind == "clean") & (g.source.isin(["dualuse", "jbb_benign"]))]
        rows.append(dict(t_block=tb, harm_medical_hard_block=(hm.action == "BLOCK").mean(),
                         harm_medical_intervene=(hm.action != "ALLOW").mean(),
                         benign_hard_overblock=(bn.action == "BLOCK").mean(),
                         benign_friction=(bn.action != "ALLOW").mean()))
    return pd.DataFrame(rows)


def latency_bench(items, ours_scorer_data, baselines, n=100, seed=0):
    clean = items[items.kind == "clean"]
    sample = clean.sample(min(n, len(clean)), random_state=seed).text.tolist()
    sc = HarmScorer(True, seed).fit(clean.text, clean.label)
    fw = InputFirewall(sc)
    rows = []
    t = [fw.inspect(x).latency_ms for x in sample]
    rows.append(dict(system="ours_full", p50_ms=np.percentile(t, 50), p95_ms=np.percentile(t, 95)))
    for name in baselines:
        try:
            m = REGISTRY[name]()
            lat = []
            for x in sample[:50]:
                t0 = time.perf_counter()
                m.predict_many([x])
                lat.append((time.perf_counter() - t0) * 1000)
            rows.append(dict(system=name, p50_ms=np.percentile(lat, 50), p95_ms=np.percentile(lat, 95)))
        except Exception as e:
            print(f"[latency] skipped {name}: {e}")
    return pd.DataFrame(rows)


def plot_tradeoff(sweep_df, meas, out_png):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(sweep_df.benign_hard_overblock * 100, sweep_df.harm_medical_hard_block * 100, "o-", label="ours (sweep of t_block)")
    for _, r in sweep_df.iterrows():
        ax.annotate(f"{r.t_block:.2f}", (r.benign_hard_overblock * 100, r.harm_medical_hard_block * 100), fontsize=7)
    for system, g in meas.groupby("system"):
        if system.startswith("ours_") and system != "ours_full":
            continue
        hb = g[(g.slice == "harm_medical") & (g.metric == "hard_block_rate")].value
        ob = g[(g.slice == "benign_dualuse") & (g.metric == "over_block_rate")].value
        if len(hb) and len(ob):
            ax.scatter(ob.iloc[0] * 100, hb.iloc[0] * 100, s=60, label=system)
    ax.set_xlabel("Over-block rate on benign dual-use clinical questions (%)  [lower is better]")
    ax.set_ylabel("Hard-block rate on harmful medical requests (%)  [higher is better]")
    ax.set_title("Safety vs over-blocking trade-off")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def wide_summary(meas):
    keys = [("harm_medical", "hard_block_rate"), ("harm_medical", "intervene_rate"),
            ("harm_general_jbb", "hard_block_rate"), ("attacks_independent", "attack_success_rate"),
            ("attacks_char_synthetic", "attack_success_rate"), ("benign_jbb", "over_block_rate"),
            ("benign_dualuse", "over_block_rate"), ("benign_dualuse", "friction_rate"),
            ("benign_easy_medquad", "over_block_rate"), ("overall_clean_strict", "f1"),
            ("overall_clean_strict", "balanced_acc")]
    rows = []
    for system, g in meas.groupby("system"):
        r = {"system": system}
        for sl, mt in keys:
            s = g[(g.slice == sl) & (g.metric == mt)]
            if len(s):
                x = s.iloc[0]
                r[f"{sl}:{mt}"] = fmt(x.value, x.lo, x.hi, int(x.n)) if x.lo == x.lo else f"{x.value:.3f}"
            else:
                r[f"{sl}:{mt}"] = "n/a"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_char", type=int, default=150)
    ap.add_argument("--baselines", default="keyword,protectai")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)

    items = build_items(Path(a.data_dir), a.quick, a.n_char, a.seed)
    print(items.groupby(["kind", "source"]).size().to_string())

    pred, sweep, items = ours_cv(items, a.folds, a.seed)
    items = items[items.fold.notna()]
    names = [b for b in a.baselines.split(",") if b]
    all_pred = [pred]
    for b in names:
        try:
            acts = run_cached(b, items.text.tolist(), out)
            all_pred.append(pd.DataFrame(dict(item_id=items.item_id.values, system=b, action=acts,
                                              risk=np.nan, route_text="", route_sys="")))
        except Exception as e:
            print(f"[warn] baseline '{b}' skipped: {type(e).__name__}: {str(e)[:200]}")
    P = pd.concat(all_pred, ignore_index=True)
    df = P.merge(items.drop(columns=["fold"]), on="item_id")

    meas = metrics_long(df)
    summary = wide_summary(meas)
    mc = mcnemar_table(df)
    sw = sweep_table(sweep, items)
    P.to_csv(out / "predictions.csv", index=False)
    items.to_csv(out / "items.csv", index=False)
    meas.to_csv(out / "metrics_long.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    mc.to_csv(out / "mcnemar.csv", index=False)
    sw.to_csv(out / "threshold_sweep.csv", index=False)
    plot_tradeoff(sw, meas, out / "tradeoff.png")
    if a.latency:
        lat = latency_bench(items, None, names)
        lat.to_csv(out / "latency.csv", index=False)
        print(lat.to_string(index=False))
    meta = dict(date=time.strftime("%Y-%m-%d %H:%M"), python=sys.version.split()[0], platform=platform.platform(),
                seed=a.seed, folds=a.folds, quick=a.quick, baselines=names, n_items=len(items),
                thresholds=dict(t_sanitize=0.35, t_block=0.70, risk='noisy-or(p_ml, rule_score)'))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 34)
    print("\n=== SUMMARY (rate % [95% Wilson CI]) ===")
    print(summary.T.to_string(header=False))
    print("\n=== McNemar (ours_full vs others, strict correctness) ===")
    print(mc[mc.subset.isin(["all_items", "benign_clean", "harmful_clean"])].to_string(index=False))
    print(f"\nSaved everything in {out}/")


if __name__ == "__main__":
    main()
