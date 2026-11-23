#!/usr/bin/env python3
"""Download the three starter datasets for the ccRCC multi-omics study.

    python download_data.py --out ccRCC_data            # all three
    python download_data.py --out ccRCC_data --only cptac tcga

Datasets
  cptac : CPTAC ccRCC processed tables (clinical, mutations, CNV, RNA,
          proteome, phosphoproteome) through the `cptac` Python package.
  tcga  : TCGA-KIRC PanCancer Atlas bundle from the cBioPortal DataHub.
  camp  : Cancer Atlas of Metabolic Profiles, Zenodo DOI 10.5281/zenodo.7150252.

Nothing is hard-coded about file names inside the repositories: the CPTAC
tables are whatever the package lists, and the Zenodo files are whatever the
record lists. Every saved file is recorded in <out>/MANIFEST.tsv with its size
and SHA-256 so the exact inputs of the study can be reported.
"""
import argparse
import hashlib
import sys
import tarfile
import traceback
from pathlib import Path

import requests

TCGA_STUDY = "kirc_tcga_pan_can_atlas_2018"
TCGA_URL = f"https://datahub.assets.cbioportal.org/{TCGA_STUDY}.tar.gz"
CAMP_DOI = "https://doi.org/10.5281/zenodo.7150252"
UA = {"User-Agent": "ccrcc-multiomics-downloader/1.0"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest):
    """Stream a URL to disk; skip if the file is already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers=UA, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        if dest.exists() and total and dest.stat().st_size == total:
            print(f"  already present: {dest.name}")
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        done = 0
        with open(tmp, "wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
                done += len(block)
                if total:
                    print(f"\r  {dest.name}: {done / total:6.1%}", end="")
        print()
        tmp.replace(dest)
    return dest


def get_cptac(out):
    """Save every table the cptac package offers for ccRCC as .csv.gz."""
    try:
        import cptac
    except ImportError:
        sys.exit("Run first:  pip install cptac pandas requests")
    d = out / "cptac_ccrcc"
    d.mkdir(parents=True, exist_ok=True)
    print(f"cptac package version {cptac.version()}")
    cc = cptac.Ccrcc()
    listing = cc.list_data_sources()
    listing.to_csv(d / "_available_tables.tsv", sep="\t", index=False)
    print(listing.to_string())
    for _, row in listing.iterrows():
        dtype, sources = row.iloc[0], row.iloc[1]
        if isinstance(sources, str):
            sources = [s.strip() for s in sources.split(",")]
        for src in sources:
            target = d / f"{dtype}__{src}.csv.gz"
            if target.exists():
                continue
            try:
                df = cc.get_dataframe(dtype, src)
                df.to_csv(target, compression="gzip")
                print(f"  saved {target.name}  shape={df.shape}")
            except Exception as exc:  # keep going; report at the end
                print(f"  FAILED {dtype}/{src}: {exc}")
                (d / f"{dtype}__{src}.FAILED.txt").write_text(traceback.format_exc())


def get_tcga(out):
    d = out / "tcga_kirc"
    tar = fetch(TCGA_URL, d / f"{TCGA_STUDY}.tar.gz")
    if not (d / TCGA_STUDY).exists():
        print("  extracting ...")
        with tarfile.open(tar) as tf:
            base = d.resolve()
            for m in tf.getmembers():  # refuse paths that escape the folder
                if not (base / m.name).resolve().is_relative_to(base):
                    sys.exit(f"unsafe path in archive: {m.name}")
            tf.extractall(d)


def get_camp(out):
    d = out / "camp_zenodo"
    final = requests.get(CAMP_DOI, headers=UA, timeout=60, allow_redirects=True).url
    rec_id = final.rstrip("/").split("/")[-1]
    meta = requests.get(f"https://zenodo.org/api/records/{rec_id}", headers=UA, timeout=60)
    meta.raise_for_status()
    meta = meta.json()
    print(f"Zenodo record {rec_id}: {meta['metadata']['title']}")
    for f in meta["files"]:
        fetch(f["links"]["self"], d / f["key"])


def write_manifest(out):
    rows = ["path\tbytes\tsha256"]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.tsv" and not p.name.endswith(".part"):
            rows.append(f"{p.relative_to(out).as_posix()}\t{p.stat().st_size}\t{sha256(p)}")
    (out / "MANIFEST.tsv").write_text("\n".join(rows) + "\n")
    print(f"Manifest: {len(rows) - 1} files -> {out / 'MANIFEST.tsv'}")


def main():
    steps = {"cptac": get_cptac, "tcga": get_tcga, "camp": get_camp}
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="ccRCC_data", type=Path)
    ap.add_argument("--only", nargs="+", choices=list(steps), default=list(steps))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in args.only:
        print(f"\n=== {name} ===")
        try:
            steps[name](args.out)
        except Exception as exc:
            failed.append(name)
            print(f"!! {name} failed: {exc}")
    write_manifest(args.out)
    if failed:
        sys.exit(f"Finished with failures: {', '.join(failed)} (re-run with --only ...)")
    print("\nAll done.")


if __name__ == "__main__":
    main()
