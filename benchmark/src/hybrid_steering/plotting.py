"""Plot saved experiment points with confidence intervals on both axes."""

import csv
import math
from pathlib import Path


def matches(row: dict[str, str], filters: dict) -> bool:
    """CSV stores numbers as strings: strength '0.0' must match numeric 0."""
    for column, expected in filters.items():
        actual = row[column]
        if isinstance(expected, (int, float)):
            actual = float(actual)
        if actual != expected:
            return False
    return True


def read_axis(rows: list[dict[str, str]], settings: dict) -> tuple[list, list, list]:
    """Convert confidence bounds into distances from each point."""
    value_column, lower_column, upper_column = settings["columns"]
    scale = settings.get("scale", 1)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Axis scale must be finite and positive")

    values, lower_errors, upper_errors = [], [], []
    for row in rows:
        value = float(row[value_column])
        lower = float(row[lower_column])
        upper = float(row[upper_column])
        if not all(math.isfinite(number) for number in (value, lower, upper)):
            raise ValueError("Point and confidence bounds must be finite")
        if not lower <= value <= upper:
            raise ValueError(f"Invalid confidence interval: {lower}, {value}, {upper}")

        values.append(value * scale)
        lower_errors.append((value - lower) * scale)
        upper_errors.append((upper - value) * scale)
    return values, lower_errors, upper_errors


def pareto_indices(x: list[float], y: list[float]) -> list[int]:
    """Indices of nondominated points, ordered by X; larger X and Y are better."""
    if len(x) != len(y):
        raise ValueError("Pareto coordinates must have equal lengths")
    frontier, best_y = [], -math.inf
    for index in sorted(range(len(x)), key=lambda i: (-x[i], -y[i], i)):
        if y[index] > best_y:
            frontier.append(index)
            best_y = y[index]
    return frontier[::-1]


def plot_tradeoff(config: dict, root: Path) -> Path:
    """Read CSV, select configurations, draw points, and save only a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (root / config["input"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Input CSV is empty")

    # Select each configuration once, before creating any output.
    selected_series = []
    used_rows: set[int] = set()
    used_colors: set[str] = set()
    for series in config["series"]:
        indices = [index for index, row in enumerate(rows)
                   if matches(row, series["match"])]
        if not indices:
            raise ValueError(f"No points match: {series['label']}")
        if used_rows.intersection(indices):
            raise ValueError(f"Point appears in multiple series: {series['label']}")
        if series["color"] in used_colors:
            raise ValueError("Each legend entry needs a distinct color")
        used_rows.update(indices)
        used_colors.add(series["color"])
        selected_rows = [rows[index] for index in indices]
        x_data = read_axis(selected_rows, config["x"])
        y_data = read_axis(selected_rows, config["y"])
        if config.get("point_labels"):
            for row in selected_rows:
                if not math.isfinite(float(row[config["point_labels"]])):
                    raise ValueError("Point label strength must be finite")
        selected_series.append((series, x_data, y_data, selected_rows))
    if not selected_series:
        raise ValueError("No plot series specified")

    pareto = config.get("pareto", {})
    if not isinstance(pareto, dict):
        raise ValueError("pareto must be an object with enabled, band and optional baseline")
    baseline = []
    if pareto.get("enabled") and pareto.get("baseline") is not None:
        baseline = [row for row in rows if matches(row, pareto["baseline"])]
        if len(baseline) != 1:
            raise ValueError("Pareto baseline must match exactly one CSV row")
        read_axis(baseline, config["x"])
        read_axis(baseline, config["y"])

    large_legend = len(selected_series) > 3
    columns = 1 if max(len(series["label"]) for series, _, _, _ in selected_series) > 30 else 2
    legend_rows = math.ceil(len(selected_series) / columns)
    height = 5.7 + max(0, legend_rows - 3) * 0.25 if large_legend else 5.7
    style = {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        figure, axes = plt.subplots(figsize=(7.2, height))
        try:
            annotations = []
            # Every point has the same marker and size; only color changes.
            for series_index, (series, x_data, y_data, selected_rows) in enumerate(selected_series):
                x, x_lower, x_upper = x_data
                y, y_lower, y_upper = y_data
                axes.errorbar(
                    x, y, xerr=[x_lower, x_upper],
                    yerr=[y_lower, y_upper] if config["y"].get("show_ci", True) else None,
                    fmt="o", linestyle="none", markersize=7, capsize=3,
                    elinewidth=1.2, color=series["color"], label=series["label"],
                )
                if config.get("point_labels"):
                    for index, (px, py, row) in enumerate(zip(x, y, selected_rows)):
                        strength = float(row[config["point_labels"]])
                        midpoint = sum(config["x"].get("limits", [min(x), max(x)])) / 2
                        left = px > midpoint * config["x"].get("scale", 1)
                        offset = (9 + index % 3 * 10) * (1 if series_index % 2 == 0 else -1)
                        annotations.append(axes.annotate(f"c={strength:g}", (px, py),
                                      xytext=(-6 if left else 6, offset), textcoords="offset points",
                                      ha="right" if left else "left", va="center", fontsize=8,
                                      color=series["color"],
                                      arrowprops={"arrowstyle": "-", "color": series["color"], "linewidth": .5},
                                      bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": .5}))
                if pareto.get("enabled") and series.get("pareto", True):
                    extra_x = read_axis(baseline, config["x"])
                    extra_y = read_axis(baseline, config["y"])
                    fx, fy = x + extra_x[0], y + extra_y[0]
                    low, high = y_lower + extra_y[1], y_upper + extra_y[2]
                    indices = pareto_indices(fx, fy)
                    axes.plot([fx[i] for i in indices], [fy[i] for i in indices],
                              color=series["color"], linewidth=1.8, zorder=3)
                    if pareto.get("band", True):
                        # Join marginal Y intervals for display; not a simultaneous frontier CI.
                        axes.fill_between([fx[i] for i in indices],
                                          [fy[i] - low[i] for i in indices],
                                          [fy[i] + high[i] for i in indices],
                                          color=series["color"], alpha=0.12, linewidth=0)

            axes.set_xlabel(config["x"]["label"])
            axes.set_ylabel(config["y"]["label"])
            if "limits" in config["x"]:
                axes.set_xlim([value * config["x"].get("scale", 1)
                               for value in config["x"]["limits"]])
            if "limits" in config["y"]:
                axes.set_ylim([value * config["y"].get("scale", 1)
                               for value in config["y"]["limits"]])
            axes.grid(color="#e5e7eb", linewidth=0.6)
            axes.set_axisbelow(True)
            if large_legend:
                handles, labels = axes.get_legend_handles_labels()
                figure.legend(handles, labels, frameon=False, loc="lower left",
                              bbox_to_anchor=(0.14, 4.389 / height + 0.03),
                              ncol=columns, borderaxespad=0, fontsize=9)
            else:
                axes.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.23),
                            borderaxespad=0, fontsize=10)
            figure.text(0.5, 0.3135 / height, f"{config['model']} · {config['concept']}",
                        ha="center", fontsize=10)
            figure.text(0.5, 0.114 / height, config["interval_note"], ha="center",
                        fontsize=8, color="#4b5563")
            figure.subplots_adjust(left=0.14, right=0.96, bottom=1.14 / height,
                                   top=4.389 / height)

            if config.get("avoid_label_overlap"):
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                occupied = []
                for annotation in annotations:
                    original_x, original_y = annotation.get_position()
                    direction = -1 if original_x < 0 else 1
                    placed = False
                    for column in range(4):
                        for shift in [0] + [sign * distance for distance in range(10, 181, 10)
                                            for sign in (1, -1)]:
                            annotation.set_position((original_x + direction * column * 45,
                                                     original_y + shift))
                            annotation.update_positions(renderer)
                            box = annotation.get_bbox_patch().get_window_extent(renderer).expanded(1.1, 1.2)
                            if (axes.bbox.contains(box.x0, box.y0)
                                    and axes.bbox.contains(box.x1, box.y1)
                                    and not any(box.overlaps(previous) for previous in occupied)):
                                occupied.append(box)
                                placed = True
                                break
                        if placed:
                            break
                    if not placed:
                        annotation.set_position((original_x, original_y))

            output = (root / config["output"]).with_suffix(".png")
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=220)
        finally:
            plt.close(figure)
    return output
