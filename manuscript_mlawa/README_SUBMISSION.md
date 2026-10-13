# AudioTwin — Machine Learning with Applications submission bundle

This directory is intentionally flat because Elsevier Editorial Manager requires LaTeX submission files and figures at one folder level.

## Main files

- `main.tex` — complete manuscript using `elsarticle` (includes Appendix A, oracle equations)
- `references.bib` / `main.bbl` — bibliography (BibTeX source and compiled)
- `fig1_framework.pdf` — framework diagram (source: `fig1_framework.tex`, TikZ)
- `fig2_learning_curves.pdf` — regret versus query budget with 95% bootstrap bands
- `fig3_effects.pdf` — context and preference-family effects at T=20
- `fig4_context_family_heatmap.pdf` — context × family analysis
- PNG versions of each figure for preview in submission portals
- `highlights.txt` — journal highlights (each ≤ 85 characters)
- `COVER_LETTER.md` — cover letter
- `main.pdf` — compiled manuscript
- `make_manuscript_figures.py` — regenerates Figs 2–4 from `../publication/results/`
- `SUBMISSION_MANIFEST.txt` — upload manifest

## Compile

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript uses the standard Elsevier `elsarticle` class, `elsarticle-num` bibliography style and the `algorithm`/`algpseudocode` packages.

## Evidence trail

Every number in the manuscript can be regenerated from `../publication/data/final_v1.parquet` with `../scripts/analyze_final.py` (frozen primary and pre-specified secondary results), `../scripts/make_publication_summaries.py` (context/family summaries and figure inputs) and `../scripts/analyze_exploratory_final.py` (exploratory final-cohort analyses). Development-seed diagnostics are in `../publication/postfinal_diagnostics/`. The final analysis uses N=128 listeners with the declared infeasible-listener replacement 55 → 128.
