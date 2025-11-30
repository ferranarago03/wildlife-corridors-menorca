from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import rcParams
from matplotlib.font_manager import fontManager
from matplotlib.patches import Patch
from pyfonts import load_google_font

# Matplotlib font setup
font = load_google_font("Courier Prime", weight="regular", italic=False)
fontManager.addfont(str(font.get_file()))
rcParams.update(
    {
        "font.family": font.get_name(),
        "font.style": font.get_style(),
        "font.weight": font.get_weight(),
        "font.size": font.get_size(),
        "font.stretch": font.get_stretch(),
        "font.variant": font.get_variant(),
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "figure.titlesize": 20,
    }
)

# Centralized palette to keep maps consistent across engines.
DEFAULT_SPECIES_COLORS = {
    "oryctolagus_cuniculus": "green",
    "atelerix_algirus": "purple",
    "eliomys_quercinus": "blue",
    "martes_martes": "red",
}


@dataclass
class SolutionSummary:
    species_list: list[str]
    built_corridors: set[str]
    corridors_by_species: dict[str, set[str]]
    rehabilitated_by_species: dict[str, set[str]]
    origin_cells_by_species: dict[str, list[str]]
    species_colors: dict[str, str]


def _serialize_set_mapping(raw: Mapping[str, set[str]]) -> dict[str, list[str]]:
    return {k: sorted(v) for k, v in raw.items()}


def _deserialize_set_mapping(raw: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    return {k: set(v) for k, v in raw.items()}


def _extract_value(raw: Any) -> float:
    """Extract numeric value from solver variables or plain numbers."""
    try:
        return float(getattr(raw, "X", raw))
    except Exception:
        return 0.0


def build_solution_summary(
    species_list: list[str],
    origin_cells_by_species: dict[str, list[str]],
    built_corridors: Iterable[str],
    flow_solution: Mapping[tuple[str, str, str, str], Any],
    rehab_solution: Mapping[tuple[str, str], Any],
    species_colors: dict[str, str] | None = None,
    threshold: float = 0.5,
) -> SolutionSummary:
    """Normalize solver outputs into a reusable summary for visualization."""
    corridors_by_species: dict[str, set[str]] = {
        species: set() for species in species_list
    }
    for key, raw_val in flow_solution.items():
        if len(key) != 4:
            continue
        species, _, j, _ = key
        if _extract_value(raw_val) > threshold:
            corridors_by_species.setdefault(species, set()).add(j)

    rehabilitated_by_species: dict[str, set[str]] = {
        species: set() for species in species_list
    }
    for key, raw_val in rehab_solution.items():
        if len(key) != 2:
            continue
        species, j = key
        if _extract_value(raw_val) > threshold:
            rehabilitated_by_species.setdefault(species, set()).add(j)

    palette = species_colors or DEFAULT_SPECIES_COLORS

    return SolutionSummary(
        species_list=list(species_list),
        built_corridors=set(built_corridors),
        corridors_by_species=corridors_by_species,
        rehabilitated_by_species=rehabilitated_by_species,
        origin_cells_by_species=origin_cells_by_species,
        species_colors=palette,
    )


def _build_legend_html(
    summary: SolutionSummary, species: str | None, title: str | None
) -> str:
    legend_title = title or "Legend"

    # Build legend content items
    content_lines: list[str] = []

    if species:
        color = summary.species_colors.get(species, "green")
        content_lines.extend(
            [
                f'<p><i class="fa fa-square" style="color:{color}"></i> {species}</p>',
                f'<p><i class="fa fa-square" style="color:{color}; opacity:0.6"></i> {species} (adapted)</p>',
                '<p><i class="fa fa-square" style="color:orange"></i> Corridor used by the species</p>',
                '<p><i class="fa fa-square" style="color:#d3d3d3"></i> Corridor of other species</p>',
            ]
        )
    else:
        for sp in summary.species_list:
            color = summary.species_colors.get(sp, "green")
            content_lines.append(
                f'<p><i class="fa fa-square" style="color:{color}"></i> {sp}</p>'
            )
            content_lines.append(
                f'<p><i class="fa fa-square" style="color:{color}; opacity:0.6"></i> {sp} (adapted)</p>'
            )
        content_lines.append(
            '<p><i class="fa fa-square" style="color:orange"></i> Built corridor</p>'
        )

    content_lines.append(
        '<p><i class="fa fa-square" style="color:lightgray"></i> Other cells</p>'
    )

    content_html = "\n".join(content_lines)

    # Build full legend with collapsible functionality
    legend_html = f"""
<style>
    .map-legend {{
        position: fixed;
        top: 50px;
        left: 50px;
        background-color: white;
        border: 2px solid grey;
        z-index: 9999;
        font-size: 14px;
        padding: 10px;
        max-height: 80vh;
        overflow: hidden;
        transition: max-height 0.3s ease;
    }}
    .map-legend.collapsed {{
        max-height: 45px;
        overflow: hidden;
    }}
    .map-legend.collapsed .legend-content {{
        display: none;
    }}
    .legend-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
    }}
    .legend-toggle {{
        background: none;
        border: 1px solid #666;
        border-radius: 3px;
        cursor: pointer;
        font-size: 12px;
        padding: 2px 8px;
        margin-left: 10px;
    }}
    .legend-toggle:hover {{
        background-color: #eee;
    }}
    .legend-content {{
        max-height: 60vh;
        overflow-y: auto;
        margin-top: 5px;
    }}
    .legend-content p {{
        margin: 4px 0;
        white-space: nowrap;
    }}
    /* Auto-collapse on small screens */
    @media (max-width: 768px) {{
        .map-legend {{
            top: 10px;
            left: 10px;
            font-size: 12px;
            max-width: calc(100vw - 80px);
        }}
        .map-legend:not(.expanded) {{
            max-height: 45px;
        }}
        .map-legend:not(.expanded) .legend-content {{
            display: none;
        }}
    }}
</style>
<div class="map-legend" id="mapLegend">
    <div class="legend-header" onclick="toggleLegend()">
        <strong>{legend_title}</strong>
        <button class="legend-toggle" id="legendToggleBtn">−</button>
    </div>
    <div class="legend-content">
        {content_html}
    </div>
</div>
<script>
    function toggleLegend() {{
        var legend = document.getElementById('mapLegend');
        var btn = document.getElementById('legendToggleBtn');
        if (legend.classList.contains('collapsed')) {{
            legend.classList.remove('collapsed');
            legend.classList.add('expanded');
            btn.textContent = '−';
        }} else {{
            legend.classList.add('collapsed');
            legend.classList.remove('expanded');
            btn.textContent = '+';
        }}
    }}
    // Auto-collapse on small screens on load
    (function() {{
        if (window.innerWidth <= 768) {{
            var legend = document.getElementById('mapLegend');
            var btn = document.getElementById('legendToggleBtn');
            legend.classList.add('collapsed');
            btn.textContent = '+';
        }}
    }})();
</script>
"""
    return legend_html


def create_solution_map(
    df: gpd.GeoDataFrame,
    summary: SolutionSummary,
    species: str | None = None,
    map_center: tuple[float, float] = (39.97, 4.0460),
    zoom_start: int = 11,
    title: str | None = None,
) -> folium.Map:
    """Build a folium map either for all species or filtered to a single one."""
    folium_map = folium.Map(
        location=list(map_center), zoom_start=zoom_start, tiles="OpenStreetMap"
    )
    df = df.copy()

    # Precompute lookups
    species_using_corridor: dict[str, list[str]] = {
        cell_id: [] for cell_id in summary.built_corridors
    }
    for sp, cells in summary.corridors_by_species.items():
        for cell in cells:
            if cell in species_using_corridor:
                species_using_corridor[cell].append(sp)

    rehab_by_cell: dict[str, list[str]] = {}
    for sp, cells in summary.rehabilitated_by_species.items():
        for cell in cells:
            rehab_by_cell.setdefault(cell, []).append(sp)

    for _, row in df.iterrows():
        grid_id = row["grid_id"]
        color = "lightgray"
        fill_opacity = 0.9
        tooltip_lines = [f"Grid ID: {grid_id}"]

        if species:
            target_corridors = summary.corridors_by_species.get(species, set())
            if grid_id in target_corridors:
                color = "orange"
                tooltip_lines.append("Corridor used by target species")
            # elif grid_id in summary.built_corridors:
            #     color = "#d3d3d3"
            #     tooltip_lines.append("Corridor of other species")

            if grid_id in rehab_by_cell and species in rehab_by_cell[grid_id]:
                color = summary.species_colors.get(species, "green")
                fill_opacity = 0.6
                tooltip_lines.append("Adapted cell for the species")

            if row.get(f"has_{species}", False):
                color = summary.species_colors.get(species, color)
                tooltip_lines.append(f"Origin of: {species}")

            # other_adapted = [
            #     sp for sp in rehab_by_cell.get(grid_id, []) if sp != species
            # ]
            # if other_adapted:
            #     tooltip_lines.append("Adapted for other species:")
            #     tooltip_lines.extend(f"- {sp}" for sp in other_adapted)
        else:
            if grid_id in summary.built_corridors:
                color = "orange"

            using_species = species_using_corridor.get(grid_id, [])
            if using_species:
                tooltip_lines.append("Species using this corridor:")
                tooltip_lines.extend(f"- {sp}" for sp in using_species)

            if grid_id in rehab_by_cell:
                first_species = rehab_by_cell[grid_id][0]
                color = summary.species_colors.get(first_species, color)
                fill_opacity = 0.6
                tooltip_lines.append("Adapted for:")
                tooltip_lines.extend(f"- {sp}" for sp in rehab_by_cell[grid_id])

            for sp in summary.species_list:
                if row.get(f"has_{sp}", False):
                    color = summary.species_colors.get(sp, color)
                    tooltip_lines.append(f"Origin of: {sp}")
                    fill_opacity = 0.9

        try:
            folium.GeoJson(
                row.geometry,
                style_function=lambda _, fill_color=color, fill_alpha=fill_opacity: {
                    "fillColor": fill_color,
                    "color": "black",
                    "weight": 0.5,
                    "fillOpacity": fill_alpha,
                },
                tooltip="<br>".join(tooltip_lines),
            ).add_to(folium_map)
        except Exception:
            # Skip invalid geometries but continue rendering.
            continue

    legend_html = _build_legend_html(summary, species, title)
    folium_map.get_root().html.add_child(folium.Element(legend_html))
    return folium_map


def create_solution_pdf(
    df: gpd.GeoDataFrame,
    summary: SolutionSummary,
    output_path: str | Path,
    *,
    title: str | None = None,
    gray_scale: bool = False,
    save: bool = True,
) -> Path:
    """
    Generate a PDF with 4 maps (2x2 grid), one per species, using Matplotlib.
    """

    NON_ACTIVE_CELLS = "lightgray"
    CORRIDOR_COLOR = "#9f1853"
    ADAPT_COLOR = "#a56eff"
    ORIGIN_COLOR = "#002d9c"

    if gray_scale:
        CORRIDOR_COLOR = "#4d4d4d"
        ADAPT_COLOR = "#969696"
        ORIGIN_COLOR = "#1a1a1a"

    def _format_species_title(species_name: str) -> str:
        """Convert 'oryctolagus_cuniculus' -> 'Oryctolagus Cuniculus'."""
        return species_name.replace("_", " ").title()

    output_path = Path(output_path)

    species_list = list(summary.species_list)[:4]
    if len(species_list) != 4:
        raise ValueError(
            f"Expected 4 species for the 2x2 layout, but got {len(species_list)}."
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    # Common extent for all subplots
    minx, miny, maxx, maxy = df.total_bounds

    for ax, species in zip(axes, species_list):
        ax.set_title(_format_species_title(species))

        target_corridors = summary.corridors_by_species.get(species, set())
        rehab_cells = summary.rehabilitated_by_species.get(species, set())

        has_col = f"has_{species}"
        has_series = df[has_col] if has_col in df.columns else False

        # Boolean masks for categories
        mask_corridor = df["grid_id"].isin(target_corridors)
        mask_rehab = df["grid_id"].isin(rehab_cells)
        mask_origin = (
            has_series if isinstance(has_series, bool) else has_series.fillna(False)
        )

        mask_other = ~(mask_corridor | mask_rehab | mask_origin)

        # Plot non-active cells
        if getattr(mask_other, "any", lambda: False)():
            df[mask_other].plot(
                ax=ax,
                facecolor=NON_ACTIVE_CELLS,
                edgecolor="black",
                linewidth=0.05,
                alpha=0.1,
            )

        # Plot corridors
        if getattr(mask_corridor, "any", lambda: False)():
            df[mask_corridor].plot(
                ax=ax,
                facecolor=CORRIDOR_COLOR,
                edgecolor="black",
                linewidth=0.05,
            )

        # Plot adapted cells
        if getattr(mask_rehab, "any", lambda: False)():
            df[mask_rehab].plot(
                ax=ax,
                facecolor=ADAPT_COLOR,
                edgecolor="black",
                linewidth=0.05,
            )

        # Plot origin cells
        if getattr(mask_origin, "any", lambda: False)():
            df[mask_origin].plot(
                ax=ax,
                facecolor=ORIGIN_COLOR,
                edgecolor="black",
                linewidth=0.05,
            )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy + 0.01)
        ax.set_axis_off()

    # Common legend consistent with the colors above
    legend_elements = [
        Patch(facecolor=CORRIDOR_COLOR, edgecolor="black", label="Wildlife corridors"),
        Patch(
            facecolor=ADAPT_COLOR,
            edgecolor="black",
            alpha=0.85,
            label="Adapted habitat",
        ),
        Patch(facecolor=ORIGIN_COLOR, edgecolor="black", alpha=1.0, label="Origin"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
    )

    if title:
        fig.suptitle(title)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    if save:
        fig.savefig(output_path, format="pdf")
    # plt.close(fig)
    # return output_path


def summary_to_dict(summary: SolutionSummary) -> dict[str, Any]:
    """Convert SolutionSummary into a JSON-serializable dict."""
    return {
        "species_list": list(summary.species_list),
        "built_corridors": sorted(summary.built_corridors),
        "corridors_by_species": _serialize_set_mapping(summary.corridors_by_species),
        "rehabilitated_by_species": _serialize_set_mapping(
            summary.rehabilitated_by_species
        ),
        "origin_cells_by_species": {
            k: list(v) for k, v in summary.origin_cells_by_species.items()
        },
        "species_colors": dict(summary.species_colors),
    }


def summary_from_dict(raw: Mapping[str, Any]) -> SolutionSummary:
    """Rehydrate SolutionSummary from a dictionary representation."""
    return SolutionSummary(
        species_list=list(raw.get("species_list", [])),
        built_corridors=set(raw.get("built_corridors", [])),
        corridors_by_species=_deserialize_set_mapping(
            raw.get("corridors_by_species", {})
        ),
        rehabilitated_by_species=_deserialize_set_mapping(
            raw.get("rehabilitated_by_species", {})
        ),
        origin_cells_by_species={
            k: list(v) for k, v in raw.get("origin_cells_by_species", {}).items()
        },
        species_colors=dict(raw.get("species_colors", DEFAULT_SPECIES_COLORS)),
    )


def save_solution_summary(
    summary: SolutionSummary, path: str | Path, *, echo: bool = True
) -> Path:
    """Persist a solution summary to disk so maps can be recreated without solving."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary_to_dict(summary)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    if echo:
        print(f"✓ Solution summary saved to {path}")
    return path


def load_solution_summary(path: str | Path) -> SolutionSummary:
    """Load a previously saved solution summary."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return summary_from_dict(payload)


def compute_solution_cost(
    summary: SolutionSummary,
    df: gpd.GeoDataFrame,
    alpha: float = 0.5,
    penalty_uncovered_origin: float | None = None,
) -> dict[str, Any]:
    """Compute the cost breakdown and objective value of a solution.

    Parameters
    ----------
    summary : SolutionSummary
        The solution summary (from load_solution_summary or build_solution_summary).
    df : gpd.GeoDataFrame
        The GeoDataFrame with cost columns (cost_corridor, cost_adaptation_*, *_benefit).
    alpha : float
        The alpha parameter used in the objective function (default 0.5).
        z = alpha * total_cost - (1 - alpha) * total_benefit
    penalty_uncovered_origin : float | None
        Penalty applied per uncovered origin. Defaults to models.gurobi.constants.PENALTY_UNCOVERED_ORIGIN
        when available; no penalty is applied if None.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - 'corridor_cost': total cost of built corridors
        - 'adaptation_cost': total cost of all adaptations
        - 'adaptation_cost_by_species': dict with adaptation cost per species
        - 'total_cost': sum of corridor + adaptation costs
        - 'benefit': total benefit from adaptations
        - 'benefit_by_species': dict with benefit per species
        - 'uncovered_origins': total count of uncovered origins
        - 'uncovered_origins_by_species': mapping with uncovered origins per species
        - 'penalty_cost': total penalty charged for uncovered origins
        - 'cost_with_penalty': total_cost + penalty_cost
        - 'penalty_uncovered_origin': penalty value used
        - 'alpha': the alpha parameter used
        - 'objective': the objective function value z = alpha * (cost + penalty) - (1 - alpha) * benefit
    """
    # Fallback to the configured default penalty if none is provided.
    if penalty_uncovered_origin is None:
        try:
            from models.gurobi.constants import PENALTY_UNCOVERED_ORIGIN
        except Exception:
            PENALTY_UNCOVERED_ORIGIN = None
        penalty_uncovered_origin = PENALTY_UNCOVERED_ORIGIN

    # Build a lookup for corridor costs
    corridor_cost_lookup = dict(zip(df["grid_id"], df["cost_corridor"]))

    # Compute corridor cost
    corridor_cost = sum(
        corridor_cost_lookup.get(cell, 0.0) for cell in summary.built_corridors
    )

    # Compute adaptation costs and benefits by species
    adaptation_cost_by_species: dict[str, float] = {}
    benefit_by_species: dict[str, float] = {}

    for species, cells in summary.rehabilitated_by_species.items():
        prefix = species.split("_")[0]
        suffix = species.split("_")[1]
        cost_col = f"cost_adaptation_{prefix}"
        benefit_col = f"{suffix}_benefit"

        species_cost = 0.0
        species_benefit = 0.0
        for cell in cells:
            row = df[df["grid_id"] == cell]
            if not row.empty:
                if cost_col in df.columns:
                    species_cost += row[cost_col].values[0]
                if benefit_col in df.columns:
                    species_benefit += row[benefit_col].values[0]

        adaptation_cost_by_species[species] = species_cost
        benefit_by_species[species] = species_benefit

    total_adaptation_cost = sum(adaptation_cost_by_species.values())
    total_benefit = sum(benefit_by_species.values())
    total_cost = corridor_cost + total_adaptation_cost

    # Compute uncovered origins and associated penalty
    uncovered_by_species: dict[str, list[str]] = {}
    for species, origins in summary.origin_cells_by_species.items():
        corridors = summary.corridors_by_species.get(species, set())
        uncovered = [origin for origin in origins if origin not in corridors]
        if uncovered:
            uncovered_by_species[species] = uncovered

    total_uncovered_origins = sum(len(v) for v in uncovered_by_species.values())
    penalty_cost = (penalty_uncovered_origin or 0.0) * total_uncovered_origins
    cost_with_penalty = total_cost + penalty_cost

    # Compute objective: z = alpha * (cost + penalty) - (1 - alpha) * benefit
    objective = alpha * cost_with_penalty - (1 - alpha) * total_benefit

    return {
        "corridor_cost": corridor_cost,
        "adaptation_cost": total_adaptation_cost,
        "adaptation_cost_by_species": adaptation_cost_by_species,
        "total_cost": total_cost,
        "cost_with_penalty": cost_with_penalty,
        "penalty_cost": penalty_cost,
        "uncovered_origins": total_uncovered_origins,
        "uncovered_origins_by_species": uncovered_by_species,
        "penalty_uncovered_origin": penalty_uncovered_origin,
        "benefit": total_benefit,
        "benefit_by_species": benefit_by_species,
        "alpha": alpha,
        "objective": objective,
    }


def format_cost_table(
    cost_data: dict[str, Any],
    title: str = "Solution Cost Breakdown",
) -> HTML:
    """Format cost data as a styled HTML table for display in Jupyter/Quarto.

    Parameters
    ----------
    cost_data : dict
        Output from compute_solution_cost().
    title : str
        Title to display above the table.

    Returns
    -------
    HTML
        IPython HTML object that renders nicely in notebooks and Quarto.
    """
    corridor_cost = float(cost_data["corridor_cost"])
    adaptation_cost = float(cost_data["adaptation_cost"])
    total_cost = float(cost_data["total_cost"])
    total_benefit = float(cost_data.get("benefit", 0.0))
    alpha = float(cost_data.get("alpha", 0.5))
    objective = float(cost_data.get("objective", total_cost))
    adaptation_by_species = cost_data.get("adaptation_cost_by_species", {})
    benefit_by_species = cost_data.get("benefit_by_species", {})

    # Build species rows for costs
    species_cost_rows = ""
    for species, cost in adaptation_by_species.items():
        species_display = species.replace("_", " ").title()
        species_cost_rows += f"""
        <tr>
            <td style="padding-left: 30px; color: #555;">↳ {species_display}</td>
            <td style="text-align: right; color: #555;">{float(cost):.2f}</td>
        </tr>"""

    # Build species rows for benefits
    species_benefit_rows = ""
    for species, benefit in benefit_by_species.items():
        species_display = species.replace("_", " ").title()
        species_benefit_rows += f"""
        <tr>
            <td style="padding-left: 30px; color: #555;">↳ {species_display}</td>
            <td style="text-align: right; color: #555;">{float(benefit):.2f}</td>
        </tr>"""

    html = f"""
    <div style="margin: 15px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <h4 style="margin-bottom: 10px; color: #333;">{title}</h4>
        <table style="border-collapse: collapse; width: 100%; max-width: 450px; font-size: 14px;">
            <thead>
                <tr style="border-bottom: 2px solid #333;">
                    <th style="text-align: left; padding: 8px 12px;">Component</th>
                    <th style="text-align: right; padding: 8px 12px;">Value</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px 12px;">Corridor Cost</td>
                    <td style="text-align: right; padding: 8px 12px;">{corridor_cost:.2f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px 12px;">Adaptation Cost</td>
                    <td style="text-align: right; padding: 8px 12px;">{adaptation_cost:.2f}</td>
                </tr>
                {species_cost_rows}
                <tr style="border-bottom: 1px solid #ddd; background-color: #f9f9f9;">
                    <td style="padding: 8px 12px;"><strong>Total Cost</strong></td>
                    <td style="text-align: right; padding: 8px 12px;"><strong>{total_cost:.2f}</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px 12px;">Total Benefit</td>
                    <td style="text-align: right; padding: 8px 12px;">{total_benefit:.2f}</td>
                </tr>
                {species_benefit_rows}
                <tr style="border-top: 2px solid #333; background-color: #e8f4e8;">
                    <td style="padding: 8px 12px;"><strong>Objective (z)</strong><br><small style="color: #666;">α={alpha} · (cost + penalty) − (1−α) · benefit</small></td>
                    <td style="text-align: right; padding: 8px 12px; font-size: 16px;"><strong>{objective:.2f}</strong></td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    return HTML(html)
