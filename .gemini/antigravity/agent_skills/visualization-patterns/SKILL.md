---
name: visualization-patterns
description: Julia visualization with Plots.jl and Makie.jl.
version: 2.0.0
agents:
  primary: julia-pro
skills:
- julia-plotting
- makie-mastery
- publication-figures
- interactive-viz
allowed-tools: [Read, Write, Task, Bash]
---

# Julia Visualization Patterns

// turbo-all

# Julia Visualization Patterns

High-performance plotting for scientific computing.

---

## Strategy & Backends (Parallel)

// parallel

### Tools

| Backend | Use Case | Implementation |
|---------|----------|----------------|
| **Plots.jl (GR)** | Fast Prototyping | `using Plots; gr()` |
| **CairoMakie** | Static/Publication | `using CairoMakie` (Vector PDF) |
| **GLMakie** | Interactive/3D | `using GLMakie` (GPU Accelerated) |
| **PlotlyJS** | Web/Interactive HTML | `using Plots; plotlyjs()` |

### Core Concepts

-   **Observables**: Reactive values used in Makie (`Node(0.0)`).
-   **Layouts**: `fig[1, 2]` indexing in Makie.

// end-parallel

---

## Decision Framework

### Choosing a Library

1.  **Exploration**: Use `Plots.jl`. It works out of the box.
2.  **Publication**: Use `CairoMakie`. Pixel perfect layouts and proper font support (`Computer Modern`).
3.  **Simulation GUI**: Use `GLMakie`. High FPS 3D rendering.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Performance (Target: 60FPS)**: Avoid recreating plots in loops. Update data in place.
2.  **Clarity (Target: 100%)**: Always label axes (`xlabel!`).
3.  **Portability (Target: 100%)**: Save as script, not just Notebook state.

### Quick Reference

-   `plot(x, y, label="Data")`
-   `heatmap(matrix, c=:viridis)`
-   `save("fig.png", current_figure())`

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Slow compilation | Use `PackageCompiler` or `sysimage` for Makie. |
| Overplotting | Use `alpha=0.5` or 2D histograms. |
| Wrong Size | Specify `size=(800, 600)`. |
| Raster Text | Use PDF/SVG export for papers. |

### Viz Checklist

- [ ] Backend selected appropriately
- [ ] Axis labels and units present
- [ ] Legend readable
- [ ] Colors are accessible (`:viridis`)
- [ ] Saved in Vector format (PDF/SVG)
