---
name: python-julia-visualization
description: "Cross-language visualization mastery: Matplotlib, Seaborn, Plotly, Bokeh, Plots.jl, Makie.jl."
version: 2.2.2
agents:
  primary: scientific-data-visualization
skills:
- multi-language-viz
- dashboard-engineering
- large-data-plotting
- real-time-viz
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.jl
- file:.py
- keyword:data
- keyword:julia
- keyword:python
- keyword:visualization
- project:Project.toml
- project:pyproject.toml
- project:requirements.txt
---

# Python and Julia Visualization

// turbo-all

# Python and Julia Visualization

Choosing the right tool for the job across ecosystems.

---

## Strategy & Ecosystem (Parallel)

// parallel

### Landscape Map

| Need | Python | Julia |
|------|--------|-------|
| **Static / Paper** | Matplotlib | CairoMakie |
| **Statistical** | Seaborn | StatsPlots.jl |
| **Interactive Web** | Plotly / Bokeh | PlotlyJS.jl / WGLMakie |
| **High Performance** | Datashader | GLMakie |
| **Dashboard** | Streamlit / Dash | Genie.jl / Pluto.jl |

### Performance Tiers

1.  **Tier 1 (<1k pts)**: Anything works.
2.  **Tier 2 (100k pts)**: Matplotlib (slow), Plotly (slow).
3.  **Tier 3 (1M+ pts)**: Datashader (Rasterize), GLMakie (GPU).

// end-parallel

---

## Decision Framework

### Selection Algorithm

1.  **Q1**: Interactive?
    *   No -> Matplotlib / CairoMakie.
    *   Yes -> Q2.
2.  **Q2**: Web Browser target?
    *   Yes -> Plotly / Bokeh.
    *   No (Desktop App) -> GLMakie / PyQtGraph.
3.  **Q3**: Big Data (>1M)?
    *   Yes -> Datashader / GLMakie.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Efficiency (Target: 100%)**: Don't send 1M SVG nodes to a browser (It will crash).
2.  **Interoperability (Target: 100%)**: Use standard formats (HDF5/CSV) to share data between languages.
3.  **Aesthetics (Target: 100%)**: Default to "Tufte" minimalism.

### Quick Reference

-   **Python**: `import matplotlib.pyplot as plt`
-   **Julia**: `using Plots`

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Plotting in Loop | Create figure once, update data. |
| Blocking Main Thread | Use separate thread/process for real-time plotting. |
| Memory Leak | `plt.close(fig)` or `plt.clf()` after saving. |

### System Checklist

- [ ] Tool matches data scale
- [ ] Export format suitable (Web vs Print)
- [ ] Interactive elements responsive
- [ ] Code is modular (Data gen separated from Plotting)
