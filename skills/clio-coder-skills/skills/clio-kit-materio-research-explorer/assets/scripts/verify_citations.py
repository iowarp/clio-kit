#!/usr/bin/env python3
"""
verify_citations; resolve the citations in Materio research output against a
bibliographic registry (Crossref) to surface unresolved identities and
metadata conflicts for researcher review.

It reads LITERATURE.md and/or .bib files, extracts citation candidates
(BibTeX entries, quoted paper titles from the "Key Papers" table the
literature agent writes, and quoted titles on other table rows that carry a
4-digit year), then queries Crossref by DOI or by title and classifies each
candidate:

  VERIFIED      matched a real published work (title similar, year consistent)
  MISMATCH      a work was found but the metadata disagrees (wrong year, or
                only a weak title match); requires source inspection
  NOT_FOUND     no plausible match in Crossref; inspect the source manually
  UNVERIFIABLE  no extractable title (e.g. a placeholder like "[ML disc. 2024]")

Design notes:
  * No DOIs are assumed; the literature agent usually emits none, so title+
    year resolution is the primary path.
  * Network is OPTIONAL. `--offline` (or a failed request) falls back to
    structural checks + placeholder detection so the tool still runs in CI or
    air-gapped sessions; unresolved entries become UNVERIFIABLE (offline).
  * Crossref is free and needs no key; we identify via a descriptive
    User-Agent. HTTP 429 / 5xx responses are retried once after a short
    backoff before an entry is reported UNVERIFIABLE.

Exit status: 1 for NOT_FOUND/MISMATCH findings; 2 for incomplete input coverage.
Zero deps (Python stdlib only).

Usage (invoked from a project root, where .claude/ and .research/ live):
  python3 assets/scripts/verify_citations.py .research/LITERATURE.md
  python3 assets/scripts/verify_citations.py --offline path.bib
  python3 assets/scripts/verify_citations.py --json task-01.bib
"""
import argparse
import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CROSSREF = "https://api.crossref.org/works"
UA = "materio-verify-citations/0.1 (Materio research assistant; citation verifier)"
TITLE_MATCH = 0.75      # >= this title similarity => same paper
TITLE_WEAK = 0.50       # [WEAK, MATCH) => MISMATCH (found but doubtful)
REQ_TIMEOUT = 15
REQ_PAUSE = 0.2         # polite spacing between Crossref calls
RETRY_BACKOFF = 2.0     # seconds before the single retry on HTTP 429 / 5xx


# ---- extraction -----------------------------------------------------------

class Cite:
    __slots__ = ("title", "year", "authors", "doi", "source", "raw")

    def __init__(self, title=None, year=None, authors=None, doi=None,
                 source="", raw=""):
        self.title = title
        self.year = year
        self.authors = authors
        self.doi = doi
        self.source = source      # where in the file it came from
        self.raw = raw            # original text snippet


def _norm(s):
    """Lowercase, strip non-alphanumerics to single spaces; for comparison."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def similarity(a, b):
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# Parse BibTeX groups without importing or executing anything. Nested title
# braces and one-line entries are common in real researcher bibliographies.
def _group(text, start):
    opening = text[start]
    closing = '}' if opening == '{' else ')' if opening == '(' else '"'
    depth, index, quoted = 1, start + 1, False
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if opening == '"':
            if char == '"':
                return text[start + 1:index], index + 1
        else:
            if char == '"':
                quoted = not quoted
            if not quoted:
                if char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
                    if depth == 0:
                        return text[start + 1:index], index + 1
        index += 1
    raise ValueError("unterminated BibTeX value")


def _bibtex_entries(text):
    index = 0
    while index < len(text):
        match = re.search(r"@(\w+)\s*([{(])", text[index:])
        if not match:
            return
        start = index + match.start()
        group_start = index + match.end() - 1
        try:
            body, end = _group(text, group_start)
        except ValueError:
            return
        index = end
        if match.group(1).lower() not in {"comment", "preamble", "string"}:
            yield body, text[start:end]


def _bibtex_fields(body):
    index = body.find(',') + 1
    fields = {}
    while index > 0 and index < len(body):
        match = re.match(r"\s*,?\s*([\w-]+)\s*=\s*", body[index:])
        if not match:
            break
        name = match.group(1).lower()
        index += match.end()
        if index >= len(body):
            break
        if body[index] in '{"':
            try:
                value, index = _group(body, index)
            except ValueError:
                break
        else:
            end = body.find(',', index)
            if end == -1:
                end = len(body)
            value, index = body[index:end].strip(), end
        fields[name] = re.sub(r"[{}]", "", value).strip()
    return fields


def extract_bibtex(text, source):
    """Extract balanced BibTeX entries, preserving complete title contents."""
    cites = []
    for body, raw in _bibtex_entries(text):
        fields = _bibtex_fields(body)
        title = fields.get("title")
        if title:
            cites.append(Cite(title=title, year=fields.get("year"),
                              authors=fields.get("author"), doi=fields.get("doi"),
                              source=source, raw=raw[:160]))
    return cites


# A quoted title: straight or curly quotes wrapping 20+ non-quote characters
# (the length floor rules out short quoted phrases and acronyms).
_QUOTED = re.compile(r"[\"“]([^\"”“]{20,})[\"”]")

# A 4-digit publication year.
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# An inline citation key like [Gong 2014] or [du Plessis 2019].
_INLINE = re.compile(r"\[([A-Za-z][\w.\- ]*?\s(?:19|20)\d{2}[a-z]?)\]")

# Placeholder-looking inline keys: contain a hyphenated/lowercase filler token
# rather than a proper surname (e.g. "ML domain-discovery 2024").
_PLACEHOLDER_HINT = re.compile(r"[a-z]+-[a-z]+|\b(?:ml|opt|disc|tbd|xxx)\b",
                               re.IGNORECASE)

# Markdown table separator row: |---|:---:|
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}")

# An author list as the literature agent writes it at the head of a Key
# Papers row: "King W.E. et al.", "W.E. King and R. Smith",
# "Tang M., Pistorius P.C.", "Scipioni Bertoli U. et al.".
_INITIALS = r"(?:[A-Z]\.\s?){1,3}"
_SURNAME = r"[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)?"
_AUTHOR = rf"(?:{_INITIALS}\s*)?{_SURNAME}(?:\s+{_INITIALS})?"
_AUTHOR_LIST = re.compile(
    rf"^{_AUTHOR}(?:\s*(?:,|;|&|\band\b)\s*{_AUTHOR})*"
    rf"(?:,?\s*et\s+al\.?)?\.?$")
# ...and at least one multi-author marker or initial, so a bare capitalized
# phrase ("Additive Manufacturing") is not mistaken for a surname pair.
_AUTHOR_MARK = re.compile(r"\bet\s+al\b|\band\b|&|,|;|\b[A-Z]\.")


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_kind(header, heading):
    """
    Classify a markdown table by its header row (falling back to the section
    heading above it). The literature agent writes two tables:

      ## Key Papers               | Paper | Year | Why Important |
      ## Methodological Landscape | Approach | Strengths | Limitations | Key Papers |

    Only the first is bibliographic. The Landscape table's "Key Papers"
    column repeats titles in shorthand, usually ellipsis-truncated, which
    produced false NOT_FOUND/MISMATCH flags, so it is skipped outright.
    """
    cells = [c.lower() for c in _cells(header)]
    first = cells[0] if cells else ""
    if "approach" in first:
        return "landscape"
    if "paper" in first:
        return "papers"
    h = (heading or "").lower()
    if "landscape" in h:
        return "landscape"
    if "key papers" in h:
        return "papers"
    return "other"


def _looks_like_authors(seg):
    """True if a Key Papers first-cell prefix reads as an author list."""
    if not seg or len(seg) > 80 or re.search(r"\d", seg):
        return False
    return bool(_AUTHOR_LIST.match(seg)) and bool(_AUTHOR_MARK.search(seg))


def _row_authors(first_cell):
    """
    Author string from the text that precedes the quoted title in a Key
    Papers row, or None when that text is not an author list. Anything
    else (a journal in *italics*, a volume number, prose) yields None so the
    Crossref query is by title alone rather than polluted with junk.
    """
    seg = re.split(r"[\"“]", first_cell, 1)[0]
    seg = re.split(r"[*_(\[]|\s[—–-]\s", seg, 1)[0]
    seg = seg.strip().strip(",;:—–- ").strip()
    return seg if _looks_like_authors(seg) else None


def _titles_in(line):
    """Yield (title, truncated) for each quoted title on a line."""
    for qm in _QUOTED.finditer(line):
        raw = qm.group(1).strip()
        truncated = raw.endswith("...") or raw.endswith("…")
        title = raw.rstrip(".…").rstrip(",;:").strip()
        yield title, truncated


def _row_year(line):
    """First 4-digit year on a row, looking outside the quoted titles first."""
    outside = _QUOTED.sub(" ", line)
    ym = _YEAR.search(outside) or _YEAR.search(line)
    return ym.group(0) if ym else None


def dedup_truncated(cites):
    """Deduplicate exact identities only; a complete prefix can be another paper.

    Preserve conflicting DOI metadata as separate findings. Explicit ellipsis
    fragments are omitted by the Markdown extractor, not guessed by title length.
    """
    seen, out = set(), []
    for cite in cites:
        key = (_norm(cite.title), cite.year, (cite.doi or "").lower())
        if key not in seen:
            seen.add(key)
            out.append(cite)
    return out


def extract_markdown(text, source):
    """
    Extract citation candidates from a LITERATURE.md-style document. Only
    three places are treated as bibliographic:
      * BibTeX entries embedded in the document
      * rows of the Key Papers table (title from the quoted string; authors
        only when the row's first cell starts with an author list)
      * rows of any other table that carry a 4-digit year
    Prose lines are never scanned (quoted emphasis phrases are not titles),
    and the Methodological Landscape table is skipped. Titles that end in an
    ellipsis are omitted as incomplete fragments. Complete prefix titles remain
    distinct; exact identity duplicates are removed. Also returns inline [Author Year] keys that look
    like placeholders.
    """
    cites = list(extract_bibtex(text, source))
    without_bibtex = text
    for _, raw in _bibtex_entries(text):
        without_bibtex = without_bibtex.replace(raw, "")
    lines = without_bibtex.splitlines()
    seen_titles = {(_norm(c.title), c.year) for c in cites}
    heading, in_table, kind = None, False, "other"
    for i, line in enumerate(lines):
        if "|" not in line:
            in_table = False
            if line.lstrip().startswith("#"):
                heading = line.lstrip("# ").strip()
            continue
        if not in_table:
            in_table = True
            # A header row is one followed by a |---| separator; a table
            # without one is classified by its section heading alone.
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if _TABLE_SEP.match(nxt):
                kind = _table_kind(line, heading)
                continue
            kind = _table_kind("", heading)
        if _TABLE_SEP.match(line) or kind == "landscape":
            continue
        year = _row_year(line)
        if kind != "papers" and not year:
            continue
        authors = _row_authors(_cells(line)[0]) if kind == "papers" else None
        for title, truncated in _titles_in(line):
            key = _norm(title)
            if truncated or (key, year) in seen_titles or len(key) < 12:
                continue
            seen_titles.add((key, year))
            cites.append(Cite(title=title, year=year, authors=authors,
                              source=source, raw=line.strip()[:160]))
    return dedup_truncated(cites), _extract_inline(text, source)


def _surnames(authors):
    """
    Plausible primary-author surname tokens from a free-form author string.
    Author order is ambiguous here; tables use both "King W.E." (surname
    first) and "W.E. King" (surname last); so we return every token that
    looks like a surname (capitalized, mostly letters) and drop initials
    like "W.E.". The caller matches this set against the work's authors.
    """
    if not authors:
        return set()
    first = re.split(r"\band\b|;|&", authors)[0]
    first = re.sub(r"\bet al\.?\b", "", first, flags=re.IGNORECASE)
    out = set()
    for tok in re.split(r"[\s,]+", first):
        letters = re.sub(r"[^A-Za-z]", "", tok)
        if len(letters) < 2:
            continue
        # An initials cluster like "W.E." -> "WE" is all-caps and short; skip.
        if tok.replace(".", "").isupper() and len(letters) <= 3:
            continue
        if tok[:1].isupper():
            out.add(_norm(letters))
    return out


def _item_surnames(item):
    """Normalized family names AND their word tokens (handles compound
    surnames like 'Scipioni Bertoli' whichever way the source split them)."""
    out = set()
    for a in (item.get("author") or []):
        fam = a.get("family")
        if not fam:
            continue
        nf = _norm(fam)
        out.add(nf)
        out.update(nf.split())
    return out


def _extract_inline(text, source):
    """Return inline [Author Year]-style keys that look like placeholders
    (a non-name filler token rather than a real surname, e.g.
    "[ML domain-discovery 2024]"); these are unverifiable by construction."""
    placeholders = []
    seen = set()
    for m in _INLINE.finditer(text):
        key = m.group(1).strip()
        if key in seen:
            continue
        seen.add(key)
        if _PLACEHOLDER_HINT.search(key):
            placeholders.append(Cite(title=None, source=source,
                                     raw="[" + key + "]"))
    return placeholders


# ---- verification ---------------------------------------------------------

def _retryable(code):
    return code == 429 or 500 <= code < 600


def _get(url):
    """GET JSON. One retry after RETRY_BACKOFF on HTTP 429 / 5xx (Crossref
    rate limiting or a transient server fault); any other error propagates."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in (0, 1):
        try:
            with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if attempt == 0 and _retryable(e.code):
                time.sleep(RETRY_BACKOFF)
                continue
            raise


DOI_ABSENT = object()   # sentinel: DOI definitively not in Crossref (HTTP 404)


def crossref_by_doi(doi):
    """Return the work dict, DOI_ABSENT if Crossref lacks the DOI (other agencies may own it),
    or None on a real network error."""
    try:
        data = _get(CROSSREF + "/" + urllib.parse.quote(doi))
        return data.get("message")
    except urllib.error.HTTPError as e:
        return DOI_ABSENT if e.code == 404 else None
    except Exception:
        return None


def crossref_by_title(title, authors=None, year=None):
    q = title
    if authors:
        q = authors + " " + q
    params = {"rows": "5", "query.bibliographic": q}
    url = CROSSREF + "?" + urllib.parse.urlencode(params)
    try:
        data = _get(url)
        return data.get("message", {}).get("items", [])
    except Exception:
        return None      # None = network failure (vs [] = no results)


def _item_year(item):
    for k in ("published-print", "published-online", "issued"):
        dp = item.get(k, {}).get("date-parts") if item else None
        if dp and dp[0] and dp[0][0]:
            return str(dp[0][0])
    return None


def verify_one(c, offline):
    """Return (status, detail_dict)."""
    if not c.title:
        return "UNVERIFIABLE", {"reason": "no extractable title (placeholder)"}
    if offline:
        return "UNVERIFIABLE", {"reason": "offline mode; not resolved"}

    # DOI path
    if c.doi:
        item = crossref_by_doi(c.doi)
        if item is DOI_ABSENT:
            return "UNVERIFIABLE", {"doi": c.doi,
                                    "reason": "DOI absent from Crossref; another registration agency may own it"}
        if item is None:
            return "UNVERIFIABLE", {"reason": "network error resolving DOI"}
        rt = (item.get("title") or [""])[0]
        sim = similarity(c.title, rt)
        ry = _item_year(item)
        year_ok = not (c.year and ry) or (c.year.isdigit() and ry.isdigit() and abs(int(c.year) - int(ry)) <= 1)
        if sim >= TITLE_MATCH and year_ok:
            return "VERIFIED", {"doi": c.doi, "matched_title": rt,
                                "similarity": round(sim, 2)}
        return "MISMATCH", {"doi": c.doi, "matched_title": rt,
                            "similarity": round(sim, 2),
                            "reason": "DOI metadata needs review: weak title match or inconsistent year"}

    # Title path
    items = crossref_by_title(c.title, c.authors, c.year)
    if items is None:
        return "UNVERIFIABLE", {"reason": "network error querying Crossref"}
    best, best_sim = None, 0.0
    for it in items:
        rt = (it.get("title") or [""])[0]
        s = similarity(c.title, rt)
        if s > best_sim:
            best, best_sim = it, s
    if best is None:
        return "NOT_FOUND", {"reason": "no Crossref results"}
    rt = (best.get("title") or [""])[0]
    ry = _item_year(best)
    detail = {"matched_title": rt, "similarity": round(best_sim, 2),
              "doi": best.get("DOI"), "matched_year": ry}

    year_ok = not (c.year and ry) or (c.year.isdigit() and ry.isdigit() and abs(int(c.year) - int(ry)) <= 1)
    # Author+year corroboration rescues abbreviated titles (e.g. a source
    # that shortened "...laser powder-bed fusion" to "...LPBF"): the primary
    # author's surname appearing in the matched work, with a consistent year,
    # is strong independent evidence it is the same paper.
    author_ok = bool(_surnames(c.authors) & _item_surnames(best))

    if best_sim >= TITLE_MATCH:
        if not year_ok:
            detail["reason"] = f"year {c.year} vs Crossref {ry}"
            return "MISMATCH", detail
        return "VERIFIED", detail
    if best_sim >= TITLE_WEAK:
        if author_ok and year_ok:
            detail["reason"] = "abbreviated title; confirmed by author + year"
            return "VERIFIED", detail
        detail["reason"] = "only a weak title match; verify manually"
        return "MISMATCH", detail
    detail["reason"] = "closest Crossref result is unrelated"
    return "NOT_FOUND", detail


# ---- driver ---------------------------------------------------------------

def collect(paths):
    cites, placeholders = [], []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"WARN: cannot read {p}: {e}", file=sys.stderr)
            continue
        if p.endswith(".bib"):
            cites += extract_bibtex(text, p)
        else:
            md_cites, md_ph = extract_markdown(text, p)
            cites += md_cites
            placeholders += md_ph
    return dedup_truncated(cites), placeholders


def main():
    ap = argparse.ArgumentParser(description="Verify citations against Crossref.")
    ap.add_argument("paths", nargs="+", help="LITERATURE.md and/or .bib files")
    ap.add_argument("--offline", action="store_true",
                    help="skip network; structural + placeholder checks only")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    unreadable = []
    for path in args.paths:
        try:
            with open(path, encoding="utf-8") as source:
                source.read()
        except (OSError, UnicodeError) as error:
            unreadable.append({"file": path, "reason": str(error)})
    readable = [path for path in args.paths if path not in {item["file"] for item in unreadable}]
    cites, placeholders = collect(readable)

    results = []
    for c in cites:
        status, detail = verify_one(c, args.offline)
        results.append((c, status, detail))
        if not args.offline and not c.doi:
            time.sleep(REQ_PAUSE)
    for ph in placeholders:
        results.append((ph, "UNVERIFIABLE",
                        {"reason": "self-flagged placeholder citation key"}))

    counts = {}
    for _, status, _ in results:
        counts[status] = counts.get(status, 0) + 1

    if args.json:
        out = {
            "coverage": {"files_requested": len(args.paths), "files_read": len(readable), "unreadable": unreadable, "candidates": len(cites)},
            "summary": counts,
            "results": [
                {"title": c.title, "year": c.year, "doi": c.doi,
                 "source": c.source, "raw": c.raw,
                 "status": status, "detail": detail}
                for c, status, detail in results
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        _print_report(results, counts, args.offline)

    bad = counts.get("NOT_FOUND", 0) + counts.get("MISMATCH", 0)
    if unreadable or not results:
        if not args.json:
            print("Citation coverage incomplete: unreadable inputs or no extractable citations", file=sys.stderr)
        return 2
    return 1 if bad else 0


_ICON = {"VERIFIED": "OK  ", "MISMATCH": "WARN", "NOT_FOUND": "FAIL",
         "UNVERIFIABLE": "??  "}


def _print_report(results, counts, offline):
    order = ["NOT_FOUND", "MISMATCH", "UNVERIFIABLE", "VERIFIED"]
    for status in order:
        group = [(c, d) for c, s, d in results if s == status]
        if not group:
            continue
        print(f"\n{status} ({len(group)}):")
        for c, d in group:
            label = c.title or c.raw or "?"
            line = f"  [{_ICON[status]}] {label[:88]}"
            print(line)
            bits = []
            if d.get("reason"):
                bits.append(d["reason"])
            if d.get("matched_title"):
                tag = "closest:" if status == "NOT_FOUND" else "~"
                bits.append(f"{tag} {d['matched_title'][:70]}")
            if d.get("doi") and status != "UNVERIFIABLE":
                bits.append(f"doi:{d['doi']}")
            if bits:
                print("           " + " | ".join(bits))
    mode = " (offline)" if offline else ""
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) \
        or "no citations found"
    print(f"\nverify-citations{mode}: {counts_str}")
    bad = counts.get("NOT_FOUND", 0) + counts.get("MISMATCH", 0)
    if bad:
        print(f"  -> {bad} citation(s) need attention.")


if __name__ == "__main__":
    sys.exit(main())
