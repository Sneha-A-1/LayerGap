"""Fetch reference metadata from Crossref for a list of DOIs and write references.bib.
Every entry is generated from the Crossref record, so titles, authors, journals and years are
those registered by the publisher."""
import json, re, sys, time, urllib.request, urllib.parse
REFS = [l.split("|") for l in open(sys.argv[1]).read().strip().splitlines() if l.strip() and not l.startswith("#")]
out, report = [], []
def clean(s): return re.sub(r"<[^>]+>", "", s or "").replace("&amp;", "&").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#","\\#")
for key, doi, expect in REFS:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi.strip())
    try:
        m = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "bibcheck/1.0 (mailto:none@example.org)"}), timeout=30))["message"]
    except Exception as e:
        report.append(f"FAIL\t{key}\t{doi}\t{e}"); continue
    title = clean((m.get("title") or [""])[0])
    ok = all(w.lower() in title.lower() for w in expect.split(";") if w)
    people = [a for a in m.get("author", []) if a.get("family")]
    groups = [a for a in m.get("author", []) if not a.get("family")]
    auth = " and ".join([f"{clean(a.get('family',''))}, {clean(a.get('given',''))}" for a in people] or
                        ["{" + clean(a.get("name", "")) + "}" for a in groups])
    yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    jr = clean((m.get("container-title") or [""])[0])
    typ = m.get("type")
    fields = {"author": auth, "title": "{" + title + "}", "year": str(yr), "doi": doi.strip()}
    if typ in ("journal-article",):
        et = "article"; fields.update({"journal": jr, "volume": m.get("volume", ""), "number": m.get("issue", ""), "pages": (m.get("page") or m.get("article-number") or "").replace("-", "--")})
    elif typ in ("proceedings-article", "book-chapter"):
        et = "inproceedings"; fields.update({"booktitle": jr, "pages": (m.get("page") or "").replace("-", "--")})
    else:
        et = "article"; fields.update({"journal": jr})
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    out.append(f"@{et}{{{key},\n{body}\n}}\n")
    report.append(f"{'OK' if ok else 'CHECK'}\t{key}\t{yr}\t{jr[:40]}\t{title[:110]}")
    time.sleep(0.2)
open(sys.argv[2], "w").write("\n".join(out))
print("\n".join(report))
