# AudioTwin — Machine Learning with Applications submission bundle

This directory is intentionally flat because Elsevier Editorial Manager requires LaTeX submission files and figures to be available at one folder level.

## Main files

- `main.tex` — complete manuscript using `elsarticle`
- `references.bib` — BibTeX database
- `fig1_learning_curves.pdf` — query-budget learning curves
- `fig2_context_effects.pdf` — context breakdown at T=20
- `fig3_family_effects.pdf` — preference-family breakdown at T=20
- `fig4_context_family_heatmap.pdf` — context × family analysis
- PNG versions of each figure for preview/use in submission portals
- `highlights.txt` — journal highlights
- `COVER_LETTER.md` — draft cover letter
- `main.pdf` — compiled and verified manuscript
- `make_manuscript_figures.py` — regenerates all manuscript figures from frozen summaries
- `SUBMISSION_MANIFEST.txt` — concise upload manifest

## Compile

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript uses the standard Elsevier `elsarticle` class and `elsarticle-num` bibliography style.

## Frozen evidence

All reported final values originate from the repository's frozen final evaluation and compact summaries under `../publication/results/`. The final analysis uses N=128 listeners and the declared infeasible-listener replacement 55 → 128.
