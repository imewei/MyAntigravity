---
name: scientific-data-visualization
description: Publication-quality figures for journals (Nature/Science/PRL).
version: 2.0.0
agents:
  primary: scientific-data-visualization
skills:
- matplotlib-mastery
- uncertainty-visualization
- latex-integration
- vector-graphics
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:data
- keyword:qa
- keyword:testing
- keyword:visualization
---

# Scientific Visualization Expert

// turbo-all

# Scientific Visualization Expert

Precision graphics for rigorous science.

---

## Strategy & Standards (Parallel)

// parallel

### Journal Requirements

| Journal | Column Width | Font | Format |
|---------|--------------|------|--------|
| **Nature** | 89 mm | Sans-serif (Helvetica) | PDF/EPS (Vector) |
| **APS (PRL)** | 3.4 in | Serif (Times) | EPS |
| **Science** | 5.5 cm | Helvetica | PDF |

### Essential Elements

-   **Uncertainty**: Error bars (`yerr`) or confidence bands (`fill_between`).
-   **Units**: SI Units in square brackets `[m/s]`.
-   **Scale**: Log-log inset if power laws involved.

// end-parallel

---

## Decision Framework

### Workflow

1.  **Compute**: Generate data -> Save to CSV/HDF5 (don't recompute for plot).
2.  **Script**: Python script loads data -> Generates Figure.
3.  **Style**: Apply `plt.style.context(['science', 'ieee'])`.
4.  **Export**: Save as PDF and PNG (preview).

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Reproducibility (Target: 100%)**: Plot script included in repo.
2.  **Honesty (Target: 100%)**: No cherry-picking data points.
3.  **Vector (Target: 100%)**: No rasterized text.

### Quick Commands

-   `plt.rcParams['font.family'] = 'serif'`
-   `fig.savefig('plot.pdf', dpi=300, bbox_inches='tight')`
-   `ax.tick_params(direction='in')`

// end-parallel

---

## Quality Assurance

### Common Errors

| Error | Fix |
|-------|-----|
| Missing Units | "Time" -> "Time [s]" |
| Unreadable Font | Min font size 8pt at print scale. |
| Bitmapped Lines | Export as PDF, not PNG. |
| Overlapping Text | `constrained_layout=True`. |

### Paper Checklist

- [ ] Print physical size verified (89mm)
- [ ] Font matches manuscript body
- [ ] Error bars on all measured points
- [ ] Vector format used
- [ ] Color is distinguishable in B&W
