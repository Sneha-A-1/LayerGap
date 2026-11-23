#!/usr/bin/env python3
"""Download open-access RNA-seq and mutation files from the GDC for the
CPTAC ccRCC expanded-cohort patients (the cohort that has metabolomics).

    python download_expanded_gdc.py --out ccRCC_data/cptac_expanded_gdc

Only open-access files are requested, so no GDC token is needed. The script
writes file_index.tsv (file -> patient, sample type, data type) next to the data.
"""
import argparse, io, json, tarfile
from pathlib import Path
import requests

CASES = [
    'C3L-00165', 'C3L-00789', 'C3L-00815', 'C3L-00966', 'C3L-00968', 'C3L-00969',
    'C3L-00970', 'C3L-00976', 'C3L-00980', 'C3L-00981', 'C3L-00982', 'C3L-00991',
    'C3L-01033', 'C3L-01034', 'C3L-01102', 'C3L-01106', 'C3L-01107', 'C3L-01459',
    'C3L-01462', 'C3L-01464', 'C3L-01465', 'C3L-01470', 'C3L-01665', 'C3L-01676',
    'C3L-01681', 'C3L-01869', 'C3L-01929', 'C3L-01951', 'C3L-01952', 'C3L-01954',
    'C3L-01958', 'C3L-01963', 'C3L-01976', 'C3L-01978', 'C3L-02175', 'C3L-02176',
    'C3L-02177', 'C3L-02201', 'C3L-02202', 'C3L-02208', 'C3L-02211', 'C3L-02214',
    'C3L-02220', 'C3L-02346', 'C3L-02352', 'C3L-02364', 'C3L-02544', 'C3L-02551',
    'C3L-02553', 'C3L-02556', 'C3L-02841', 'C3L-02842', 'C3L-02856', 'C3L-02857',
    'C3N-00647', 'C3N-00715', 'C3N-00721', 'C3N-01077', 'C3N-01078', 'C3N-01080',
    'C3N-01081', 'C3N-01082', 'C3N-01083', 'C3N-01085', 'C3N-01086', 'C3N-01088',
    'C3N-01090', 'C3N-01091', 'C3N-01536', 'C3N-01654', 'C3N-01655', 'C3N-01656',
    'C3N-01657', 'C3N-01806', 'C3N-01828', 'C3N-01902', 'C3N-01905', 'C3N-01989',
    'C3N-02005', 'C3N-02015', 'C3N-02063', 'C3N-02066', 'C3N-02068', 'C3N-02070',
    'C3N-02082', 'C3N-02084', 'C3N-02085', 'C3N-02160', 'C3N-02224', 'C3N-02248',
    'C3N-02262', 'C3N-02263', 'C3N-02264', 'C3N-02265', 'C3N-02266', 'C3N-02332',
    'C3N-02338', 'C3N-02429', 'C3N-02431', 'C3N-02439', 'C3N-02577', 'C3N-02723',
    'C3N-02726', 'C3N-02761', 'C3N-02763', 'C3N-02811', 'C3N-02945', 'C3N-03018',
    'C3N-03019', 'C3N-03020',
]
API = "https://api.gdc.cancer.gov"
WANTED = ["Gene Expression Quantification", "Masked Somatic Mutation"]
FIELDS = ("file_id,file_name,data_type,file_size,analysis.workflow_type,"
          "cases.submitter_id,cases.samples.sample_type,cases.samples.submitter_id")


def query():
    flt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["CPTAC-3"]}},
        {"op": "in", "content": {"field": "cases.submitter_id", "value": CASES}},
        {"op": "in", "content": {"field": "data_type", "value": WANTED}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
    ]}
    r = requests.post(f"{API}/files", json={"filters": flt, "fields": FIELDS,
                                           "format": "JSON", "size": 5000}, timeout=120)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("ccRCC_data/cptac_expanded_gdc"))
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    hits = query()
    rows = ["file_id\tfile_name\tdata_type\tworkflow\tcase\tsample_type\tsample_id"]
    for h in hits:
        case = h["cases"][0]
        smp = case.get("samples", [{}])
        rows.append("\t".join([h["file_id"], h["file_name"], h["data_type"],
                               h.get("analysis", {}).get("workflow_type", ""), case["submitter_id"],
                               ";".join(s.get("sample_type", "") for s in smp),
                               ";".join(s.get("submitter_id", "") for s in smp)]))
    (out / "file_index.tsv").write_text("\n".join(rows) + "\n")
    by_type = {}
    for h in hits:
        by_type[h["data_type"]] = by_type.get(h["data_type"], 0) + 1
    print(f"{len(hits)} open-access files found: {by_type}")
    todo = [h for h in hits if not (out / h["file_id"] / h["file_name"]).exists()]
    for i in range(0, len(todo), 40):          # the GDC bundles each batch as one tar
        batch = todo[i:i + 40]
        r = requests.post(f"{API}/data", json={"ids": [h["file_id"] for h in batch]}, timeout=900)
        r.raise_for_status()
        if len(batch) == 1:                    # a single file is returned as-is
            d = out / batch[0]["file_id"]; d.mkdir(exist_ok=True)
            (d / batch[0]["file_name"]).write_bytes(r.content)
        else:
            with tarfile.open(fileobj=io.BytesIO(r.content)) as tf:
                base = out.resolve()
                for m in tf.getmembers():
                    if not (base / m.name).resolve().is_relative_to(base):
                        raise SystemExit(f"unsafe path in archive: {m.name}")
                tf.extractall(out)
        print(f"  downloaded {min(i + 40, len(todo))}/{len(todo)}")
    print("Done.")


if __name__ == "__main__":
    main()
