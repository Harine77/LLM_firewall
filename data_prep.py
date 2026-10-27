"""Download the free datasets and build data/prompts.csv and data/attacks.csv.

Sources (all free, no login):
  MedSafetyBench  github.com/AI4LIFE-GROUP/med-safety-bench   1,800 harmful medical requests (MIT)
  JBB-Behaviors   HuggingFace JailbreakBench/JBB-Behaviors     100 harmful + 100 benign (MIT)
  JBB artifacts   github.com/JailbreakBench/artifacts          real PAIR / GCG / JailbreakChat / random-search attacks
  MedQuAD         github.com/abachaa/MedQuAD                   ~47k benign medical questions (NIH sources)
  dual-use set    data/dualuse_benign.csv                      benign but dangerous-sounding clinical questions (yours to extend)

python data_prep.py [--n_medquad 600] [--out data]
"""
import argparse
import io
import json
import math
import random
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW = "https://raw.githubusercontent.com"
ARTIFACTS = [
    ("PAIR", "PAIR/black_box/vicuna-13b-v1.5.json"),
    ("GCG", "GCG/white_box/vicuna-13b-v1.5.json"),
    ("JBC", "JBC/manual/vicuna-13b-v1.5.json"),
    ("RS", "prompt_with_random_search/black_box/gpt-3.5-turbo-1106.json"),
]


def fetch(url, dest: Path, tries=4):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return dest
            print(f"  HTTP {r.status_code} for {url}")
        except requests.RequestException as e:
            print(f"  retry {i + 1}: {e}")
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"could not download {url}")


def load_medsafety(raw: Path):
    rows = []
    for split in ("train", "test"):
        for src in ("gpt4", "llama2"):
            for k in range(1, 10):
                name = f"med_safety_demonstrations_category_{k}.csv"
                p = fetch(f"{RAW}/AI4LIFE-GROUP/med-safety-bench/main/datasets/{split}/{src}/{name}",
                          raw / "medsafety" / split / src / name)
                df = pd.read_csv(p)
                for t in df["harmful_medical_request"].dropna():
                    rows.append((str(t).strip(), 1, "medsafety", f"msb_cat{k}", f"msb_{split}_{src}_{k}"))
    return pd.DataFrame(rows, columns=["text", "label", "source", "group", "behavior"])


def load_medquad(raw: Path, n: int, seed: int):
    z = fetch("https://codeload.github.com/abachaa/MedQuAD/zip/refs/heads/master", raw / "medquad.zip")
    zf = zipfile.ZipFile(z)
    qs = []
    for name in zf.namelist():
        if not name.endswith(".xml"):
            continue
        try:
            root = ET.fromstring(zf.read(name))
        except ET.ParseError:
            continue
        folder = name.split("/")[1]
        for qa in root.iter("QAPair"):
            q = qa.find("Question")
            if q is not None and q.text and len(q.text.strip()) > 10:
                qs.append((q.text.strip(), folder))
    df = pd.DataFrame(qs, columns=["text", "group"]).drop_duplicates("text")
    per = math.ceil(n / df["group"].nunique())
    rng = random.Random(seed)
    parts = [g.sample(min(len(g), per), random_state=rng.randint(0, 10 ** 6)) for _, g in df.groupby("group")]
    out = pd.concat(parts).head(n).copy()
    out["label"], out["source"] = 0, "medquad"
    out["group"] = "mq_" + out["group"]
    out["behavior"] = ["mq_%d" % i for i in range(len(out))]
    return out[["text", "label", "source", "group", "behavior"]]


def load_jbb(raw: Path):
    """Returns (harmful_df, benign_df_or_None, attacks_df)."""
    attacks, harm_from_artifacts = [], None
    for name, path in ARTIFACTS:
        p = fetch(f"{RAW}/JailbreakBench/artifacts/main/attack-artifacts/{path}", raw / "jbb_artifacts" / f"{name}.json")
        js = json.loads(p.read_text())["jailbreaks"]
        if name == "PAIR":
            harm_from_artifacts = pd.DataFrame(
                [(j["goal"], j["behavior"], j["category"]) for j in js], columns=["Goal", "Behavior", "Category"])
        for j in js:
            if j.get("prompt"):
                attacks.append((j["prompt"], 1, f"attack_{name}", f"jbb_{j['category']}", f"harm_{j['behavior']}", name,
                                bool(j.get("jailbroken"))))
    att = pd.DataFrame(attacks, columns=["text", "label", "source", "group", "behavior", "attack_type", "jailbroken"])

    harm, ben = None, None
    try:
        from datasets import load_dataset
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        harm, ben = ds["harmful"].to_pandas(), ds["benign"].to_pandas()
    except Exception as e:  # offline / datasets not installed
        print(f"  [warn] could not load JBB from HuggingFace ({type(e).__name__}); "
              f"harmful goals taken from the GitHub artifacts, benign JBB skipped.")
        harm = harm_from_artifacts
    hdf = pd.DataFrame({"text": harm["Goal"], "label": 1, "source": "jbb_harmful",
                        "group": "jbb_" + harm["Category"], "behavior": "harm_" + harm["Behavior"]})
    bdf = None
    if ben is not None:
        bdf = pd.DataFrame({"text": ben["Goal"], "label": 0, "source": "jbb_benign",
                            "group": "jbb_" + ben["Category"], "behavior": "ben_" + ben["Behavior"]})
    return hdf, bdf, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_medquad", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    raw = out / "raw"
    print("MedSafetyBench ...")
    ms = load_medsafety(raw)
    print("MedQuAD ...")
    mq = load_medquad(raw, a.n_medquad, a.seed)
    print("JailbreakBench ...")
    jh, jb, att = load_jbb(raw)
    du = pd.read_csv(out / "dualuse_benign.csv")
    dud = pd.DataFrame({"text": du["text"], "label": 0, "source": "dualuse",
                        "group": "du_" + du["category"], "behavior": ["du_%d" % i for i in range(len(du))]})
    parts = [ms, mq, jh, dud] + ([jb] if jb is not None else [])
    prompts = pd.concat(parts, ignore_index=True).drop_duplicates("text").reset_index(drop=True)
    prompts.insert(0, "id", ["p%05d" % i for i in range(len(prompts))])
    att = att.drop_duplicates("text").reset_index(drop=True)
    att.insert(0, "id", ["a%05d" % i for i in range(len(att))])
    prompts.to_csv(out / "prompts.csv", index=False)
    att.to_csv(out / "attacks.csv", index=False)
    print("\nprompts.csv:")
    print(prompts.groupby(["source", "label"]).size().to_string())
    print("\nattacks.csv:")
    print(att.groupby("attack_type").size().to_string())
    print(f"\nWrote {out/'prompts.csv'} and {out/'attacks.csv'}")


if __name__ == "__main__":
    main()
