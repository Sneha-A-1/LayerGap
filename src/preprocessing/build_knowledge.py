"""Build the pathway knowledge base used by SLUICE.

Sources
  * Human-GEM v2.1.1 (SysBioChalmers/Human-GEM): reaction -> subsystem, enzyme genes, metabolites.
  * MSigDB Hallmark gene sets (50 sets) for non-metabolic tumour programmes.

Output: knowledge.pkl with
  pathways : {name: {"genes": [...symbols], "mets": [...normalised names], "kind": "metabolic"|"hallmark"}}
  met_ids  : {normalised metabolite name: {"kegg": ..., "hmdb": ...}}
  edges    : DataFrame(gene, met, pathway) of enzyme-metabolite pairs sharing a reaction
"""
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

EXCLUDE_SUBSYSTEMS = ("Transport reactions", "Exchange/demand reactions", "Isolated", "Miscellaneous",
                      "Pool reactions", "Artificial reactions", "Drug metabolism", "Xenobiotics metabolism")
CURRENCY = {"h2o", "h", "atp", "adp", "amp", "nad", "nadh", "nadp", "nadph", "coa", "pi", "ppi", "o2", "co2", "fad",
            "fadh2", "h2o2", "hco3", "na", "k", "chloride", "nh4", "gtp", "gdp", "udp", "ump", "ctp", "cdp", "cmp",
            "utp", "acetylcoa", "so4", "fe2", "fe3", "ca2", "oh1", "e", "photon"}
# Trivial-name synonyms between the CPTAC metabolite table and Human-GEM (documented, applied after normalisation)
SYNONYMS = {"alphaketoglutarate": "akg", "adenosine5monophosphate": "amp", "guanosine5monophosphate": "gmp",
            "uridine5monophosphate": "ump", "cytidine5monophosphate": "cmp", "inosine5monophosphate": "imp",
            "glutathione": "gsh", "oxidizedglutathione": "gssg", "glutathionedisulfide": "gssg",
            "sadenosylmethionine": "sam", "sadenosylhomocysteine": "sah", "5smethyl5thioadenosine": "5methylthioadenosine",
            "3hydroxybutyrate": "r3hydroxybutanoate", "betahydroxybutyrate": "r3hydroxybutanoate",
            "acetylcarnitine": "oacetylcarnitine", "phosphoenolpyruvate": "pep", "nicotinamideadeninedinucleotide": "nad",
            "glucose6phosphate": "glucose6phosphate", "fructose16bisphosphate": "fructose16bisphosphate",
            "dihydroxyacetonephosphate": "dhap", "glyceraldehyde3phosphate": "gap", "3phosphoglycerate": "3phosphodglycerate",
            "6phosphogluconate": "6phosphodgluconate", "ribose5phosphate": "ribose5phosphate", "hydroxyproline": "trans4hydroxyproline",
            "4hydroxyproline": "trans4hydroxyproline", "oxoproline": "5oxoproline", "pyroglutamate": "5oxoproline",
            "cystine": "cystine", "gammaaminobutyrate": "gaba", "4aminobutyrate": "gaba", "uridinediphosphateglucose": "udpglucose",
            "udpglcnac": "udpnacetylglucosamine", "cytidine5diphosphocholine": "cdpcholine", "phosphocholine": "cholinephosphate",
            "phosphorylcholine": "cholinephosphate", "glycerol3phosphate": "snglycerol3phosphate", "betaine": "betaine",
            "nacetylaspartate": "nacetylaspartate", "argininosuccinate": "argininosuccinate"}


def norm_met(name):
    """Normalise a metabolite name so that 'L-Glutamic acid' and 'glutamate' meet."""
    s = str(name).lower().strip()
    s = re.sub(r"\[[a-z]\]$", "", s)
    s = re.sub(r"^(l|d|dl)-", "", s)
    s = re.sub(r"\b(l|d|dl)-", "", s)
    s = s.replace("ic acid", "ate").replace("oic acid", "oate")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return SYNONYMS.get(s, s)


def parse_human_gem(yml_path):
    """Line parser for the Human-GEM YAML (fast, avoids loading 7 MB through PyYAML)."""
    mets, rxns = {}, []
    section, cur, sub = None, None, None
    with open(yml_path) as fh:
        for line in fh:
            if line.startswith("- metabolites:"):
                section = "m"; continue
            if line.startswith("- reactions:"):
                section = "r"; continue
            if line.startswith("- genes:"):
                break
            s = line.strip()
            if section == "m":
                if s.startswith("- id:"):
                    cur = s.split(": ", 1)[1]
                elif s.startswith("- name:") and cur:
                    mets[cur] = s.split(": ", 1)[1].strip('"')
            elif section == "r":
                if s == "- !!omap":
                    cur = {"mets": [], "genes": [], "subs": []}; rxns.append(cur); sub = None
                elif s.startswith("- metabolites:"):
                    sub = "mets"
                elif s.startswith("- gene_reaction_rule:"):
                    cur["genes"] = re.findall(r"ENSG\d+", s); sub = None
                elif s.startswith("- subsystem:"):
                    sub = "subs"
                elif re.match(r"- [A-Za-z_]+:", s) and not s.startswith("- MAM"):
                    sub = None
                elif sub == "mets" and s.startswith("- MAM"):
                    cur["mets"].append(s[2:].split(":")[0])
                elif sub == "subs" and s.startswith("- "):
                    cur["subs"].append(s[2:].strip())
    return mets, rxns


def main(src_dir, out_path):
    src_dir = Path(src_dir)
    mets, rxns = parse_human_gem(src_dir / "Human-GEM.yml")
    genes = pd.read_csv(src_dir / "genes.tsv", sep="\t").set_index("genes")["geneSymbols"].dropna().to_dict()
    mt = pd.read_csv(src_dir / "metabolites.tsv", sep="\t")
    mt["norm"] = mt["mets"].map(mets).map(norm_met)
    met_ids = {r.norm: {"kegg": r.metKEGGID, "hmdb": r.metHMDBID} for r in mt.itertuples()}

    currency = set(CURRENCY)

    pw = defaultdict(lambda: {"genes": set(), "mets": set(), "kind": "metabolic"})
    edges = set()
    for r in rxns:
        gs = {genes[g] for g in r["genes"] if g in genes}
        ms = {norm_met(mets.get(x, x)) for x in r["mets"]} - currency
        for sname in r["subs"]:
            if sname.startswith(EXCLUDE_SUBSYSTEMS):
                continue
            pw[sname]["genes"] |= gs
            pw[sname]["mets"] |= ms
            for g in gs:
                for m in ms:
                    edges.add((g, m, sname))
    with open(src_dir / "h.gmt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            pw[p[0]] = {"genes": set(p[2:]), "mets": set(), "kind": "hallmark"}
    pathways = {k: {"genes": sorted(v["genes"]), "mets": sorted(v["mets"]), "kind": v["kind"]}
                for k, v in pw.items() if len(v["genes"]) >= 5}
    out = {"pathways": pathways, "met_ids": met_ids, "currency": sorted(currency),
           "edges": pd.DataFrame(sorted(edges), columns=["gene", "met", "pathway"]),
           "versions": {"Human-GEM": "2.1.1", "MSigDB Hallmark": "v7.0"}}
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh)
    n_met = sum(v["kind"] == "metabolic" for v in pathways.values())
    print(f"pathways: {len(pathways)} ({n_met} metabolic, {len(pathways) - n_met} hallmark); "
          f"enzyme-metabolite edges: {len(edges)}; currency metabolites removed: {len(currency)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
