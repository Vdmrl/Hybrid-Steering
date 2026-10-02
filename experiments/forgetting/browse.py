"""Write one HTML page per concept with every generation from decay and scale runs."""

import argparse
import json
from html import escape
from pathlib import Path

from hybrid_steering.runtime import import_path, read_jsonl

meta_kind = import_path(Path(__file__).with_name("meta.py")).meta_kind

KEEP = ("method", "scale", "filler", "length", "index", "response", "repetition", "hit")
KEEP += ("concept_score", "content_quality", "label", "reason", "flags", "evaluable")

PAGE = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:0;background:#f6f6f4;color:#1d1d1b}}
header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:1}}
h1{{font-size:17px;margin:0 0 8px}}
.filters{{display:flex;flex-wrap:wrap;gap:8px;align-items:end}}
label{{display:flex;flex-direction:column;font-size:11px;color:#666}}
select,input{{font:13px system-ui;padding:3px 5px}}
#stats{{margin-top:8px;font-size:13px;color:#333}}
main{{padding:12px 16px;max-width:1100px}}
.card{{background:#fff;border:1px solid #ddd;border-left:4px solid #bbb;border-radius:4px;margin:0 0 10px;padding:8px 12px}}
.card.hit{{border-left-color:#2a9d8f}}
.meta{{font-size:12px;color:#555;display:flex;flex-wrap:wrap;gap:10px}}
.meta b{{color:#111}}
.q{{font-weight:600;margin:6px 0 4px}}
pre{{white-space:pre-wrap;font:13px/1.4 ui-monospace,monospace;margin:4px 0;background:#fafaf8;padding:6px;border-radius:3px}}
.reason{{font-size:12px;color:#555;font-style:italic}}
.flag{{background:#fde8e8;color:#9b1c1c;border-radius:3px;padding:0 4px}}
button{{font:13px system-ui;padding:3px 10px}}
</style>
<header><h1>{title}</h1><div class="filters" id="filters"></div><div id="stats"></div></header>
<main id="list"></main>
<script>
const DATA = {data};
const Q = DATA.questions, ROWS = DATA.rows, FIELDS = ["run","method","scale","filler","length","hit","meta","index"];
const state = {{page:0, order:null}};
const box = document.getElementById("filters");
const values = f => [...new Set(ROWS.map(r => r[f]))].sort((a,b) => typeof a === "number" ? a-b : String(a).localeCompare(String(b)));
for (const f of FIELDS) {{
  const l = document.createElement("label"); l.textContent = f === "index" ? "question" : f;
  const s = document.createElement("select"); s.id = "f-" + f;
  s.innerHTML = "<option value=''>all</option>" + values(f).map(v => `<option>${{v}}</option>`).join("");
  s.onchange = () => {{state.page = 0; render();}}; l.appendChild(s); box.appendChild(l);
}}
box.insertAdjacentHTML("beforeend", "<label>search<input id='search' size=24></label>"
  + "<button id='shuffle'>shuffle</button><button id='sorted'>sorted</button><button id='prev'>&lt;</button><button id='next'>&gt;</button>");
document.getElementById("search").oninput = () => {{state.page = 0; render();}};
document.getElementById("shuffle").onclick = () => {{state.order = ROWS.map(() => Math.random()); state.page = 0; render();}};
document.getElementById("sorted").onclick = () => {{state.order = null; state.page = 0; render();}};
document.getElementById("prev").onclick = () => {{state.page = Math.max(0, state.page-1); render();}};
document.getElementById("next").onclick = () => {{state.page++; render();}};
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));
function render() {{
  const want = Object.fromEntries(FIELDS.map(f => [f, document.getElementById("f-"+f).value]));
  const text = document.getElementById("search").value.toLowerCase();
  let ids = ROWS.map((_, i) => i).filter(i => {{
    const r = ROWS[i];
    if (FIELDS.some(f => want[f] !== "" && String(r[f]) !== want[f])) return false;
    return !text || (r.response + " " + Q[r.index] + " " + (r.reason||"")).toLowerCase().includes(text);
  }});
  if (state.order) ids.sort((a,b) => state.order[a] - state.order[b]);
  const n = ids.length, hits = ids.filter(i => ROWS[i].hit).length;
  const quality = ids.map(i => ROWS[i].content_quality).filter(v => v != null);
  const pages = Math.max(1, Math.ceil(n/50)); state.page = Math.min(state.page, pages-1);
  document.getElementById("stats").textContent = `${{n}} rows · hit rate ${{n ? (hits/n).toFixed(2) : "–"}}`
    + ` · mean quality ${{quality.length ? (quality.reduce((a,b)=>a+b,0)/quality.length).toFixed(2) : "–"}}`
    + ` · page ${{state.page+1}}/${{pages}}`;
  document.getElementById("list").innerHTML = ids.slice(state.page*50, state.page*50+50).map(i => {{
    const r = ROWS[i];
    return `<div class="card ${{r.hit ? "hit" : ""}}"><div class="meta"><b>${{esc(r.run)}}</b><span>method <b>${{r.method}}</b></span>`
      + `<span>scale <b>${{r.scale}}</b></span><span>filler <b>${{r.filler}}</b></span><span>L <b>${{r.length}}</b></span>`
      + `<span>hit <b>${{r.hit}}</b></span><span>concept <b>${{r.concept_score ?? r.label ?? "–"}}</b></span>`
      + `<span>quality <b>${{r.content_quality ?? "–"}}</b></span><span>repetition <b>${{(r.repetition||0).toFixed(2)}}</b></span>`
      + (r.meta !== "none" ? `<span class="flag">meta: ${{r.meta}}</span>` : "")
      + (r.evaluable === false ? `<span class="flag">unevaluable</span>` : "")
      + (r.flags||[]).map(f => `<span class="flag">${{esc(f)}}</span>`).join("") + `</div>`
      + `<div class="q">Q${{r.index}}: ${{esc(Q[r.index])}}</div><pre>${{esc(r.response)}}</pre>`
      + (r.reason ? `<div class="reason">${{esc(r.reason)}}</div>` : "") + `</div>`;
  }}).join("") || "<p>No rows match.</p>";
}}
render();
</script>
"""


def runs(root: Path, slug: str) -> list[tuple[str, Path]]:
    found = []
    for base, prefix in ((root / slug, "decay"), (root / "scale" / slug, "scale")):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("rows.jsonl")):
            name = path.parent.relative_to(base).as_posix()
            found.append((f"{prefix}/{'grid' if name == '.' else name}", path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/forgetting"))
    parser.add_argument("--output", type=Path, default=Path("runs/forgetting/generations"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    skip = {"scale", "generations", "figures"}
    slugs = sorted(
        p.name
        for p in args.runs.iterdir()
        if p.is_dir() and p.name not in skip and any(p.glob("*"))
    )
    links = []
    for slug in slugs:
        questions: dict[str, int] = {}
        rows = []
        for run, path in runs(args.runs, slug):
            for row in read_jsonl(path):
                key = f"{row.get('split')}:{row['question']}"
                row["index"] = questions.setdefault(key, len(questions))
                rows.append(
                    {
                        "run": run,
                        "meta": meta_kind(row["response"]) or "none",
                        **{k: row[k] for k in KEEP if k in row},
                    }
                )
        text = [q.split(":", 1)[1] for q in questions]
        data = json.dumps({"questions": text, "rows": rows}, ensure_ascii=False)
        page = PAGE.format(title=escape(f"{slug} generations"), data=data.replace("</", "<\\/"))
        (args.output / f"{slug}.html").write_text(page)
        links.append(f"<li><a href='{slug}.html'>{escape(slug)}</a> ({len(rows)} rows)</li>")
        print(f"{slug}: {len(rows)} rows", flush=True)
    (args.output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Generations</title>"
        "<h1>Generations by concept</h1><ul>" + "".join(links) + "</ul>"
    )


if __name__ == "__main__":
    main()
