"""Summarize scored generation rows as an HTML table.

Rows are plain JSON objects. ``x`` and an optional ``series`` pick the groups.
``value`` is averaged. The table does not know which experiment produced the rows.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from .runtime import read_jsonl


def group_mean(rows: list[dict], x: str, value: str, series: str | None = None) -> list[dict]:
    """Average ``value`` for each ``x``, and for each series when one is named."""
    if not rows:
        raise ValueError("rows must not be empty")
    totals: dict[tuple, list[float]] = {}
    for row in rows:
        raw = _field(row, value)
        if raw is None:
            continue
        key = (_field(row, x), _field(row, series) if series else None)
        totals.setdefault(key, []).append(float(raw))
    if not totals:
        raise ValueError(f"no numeric {value}")
    grouped = []
    for (x_value, series_value), samples in totals.items():
        item = {x: x_value, value: sum(samples) / len(samples), "n": len(samples)}
        if series:
            item[series] = series_value
        grouped.append(item)
    return grouped


def summary_section(
    rows: list[dict],
    x: str,
    value: str,
    series: str | None = None,
    title: str = "",
) -> str:
    """HTML table of the grouped mean. A series becomes the row axis."""
    grouped = group_mean(rows, x, value, series)
    heading = f"<h2>{escape(title or value)}</h2>"
    if series is None:
        return heading + _bars(grouped, x, value)
    return heading + _grid(grouped, x, series, value)


def write_page(path: str | Path, title: str, sections: list[str]) -> None:
    """Write one HTML page from already rendered sections."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = (
        "<!doctype html><meta charset=utf-8>"
        f"<title>{escape(title)}</title>"
        "<style>body{font:15px sans-serif;margin:24px;max-width:960px}"
        "table{border-collapse:collapse;margin:12px 0}"
        "td,th{border:1px solid #ccc;padding:6px 8px;text-align:right}"
        "td.bar{text-align:left;min-width:160px}</style>"
        f"<h1>{escape(title)}</h1>" + "\n".join(sections)
    )
    path.write_text(document, encoding="utf-8")


def report_main(title: str, sections: list[tuple[str, str, str | None, str]]) -> None:
    """CLI: ``--rows`` JSONL and ``--output`` HTML. Each section is x, value, series, title."""
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.rows)
    rendered = [
        summary_section(rows, x, value, series, heading)
        for x, value, series, heading in sections
        if rows and value in rows[0]
    ]
    if not rendered:
        raise ValueError("rows have none of the requested columns")
    write_page(args.output, title, rendered)
    print(f"wrote {args.output}", flush=True)


def _field(row: dict, name: str):
    if name not in row:
        raise ValueError(f"row is missing {name}")
    return row[name]


def _order(values: list) -> list:
    return sorted(set(values), key=_sort_key)


def _sort_key(value) -> tuple:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _units(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [0.0 if high == 0 else 1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _shade(unit: float) -> str:
    channel = int(255 - unit * 160)
    return f"rgb({channel},{channel},255)"


def _bars(grouped: list[dict], x: str, value: str) -> str:
    ordered = sorted(grouped, key=lambda item: _sort_key(item[x]))
    units = _units([item[value] for item in ordered])
    body = []
    for item, unit in zip(ordered, units, strict=True):
        width = int(unit * 100)
        body.append(
            "<tr>"
            f"<td>{escape(str(item[x]))}</td>"
            f"<td>{item[value]:.3f}</td>"
            f"<td>{item['n']}</td>"
            f'<td class="bar"><div style="width:{width}%;background:{_shade(unit)}">&nbsp;</div></td>'
            "</tr>"
        )
    return (
        f"<table><tr><th>{escape(x)}</th><th>{escape(value)}</th><th>n</th><th></th></tr>"
        + "".join(body)
        + "</table>"
    )


def _grid(grouped: list[dict], x: str, series: str, value: str) -> str:
    columns = _order([item[x] for item in grouped])
    rows = _order([item[series] for item in grouped])
    cells = {(item[series], item[x]): item for item in grouped}
    units = _units([item[value] for item in grouped])
    scale = {(item[series], item[x]): unit for item, unit in zip(grouped, units, strict=True)}
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        cells_html = []
        for column in columns:
            item = cells.get((row, column))
            if item is None:
                cells_html.append("<td></td>")
                continue
            unit = scale[(row, column)]
            cells_html.append(f'<td style="background:{_shade(unit)}">{item[value]:.3f}</td>')
        body.append(f"<tr><th>{escape(str(row))}</th>{''.join(cells_html)}</tr>")
    return f"<table><tr><th>{escape(series)}</th>{header}</tr>" + "".join(body) + "</table>"
