# dl-jmp/paper: Job-market paper outline

**Status:** skeleton/outline only. The model equations are provisional, the results section contains no numbers, and the BibTeX entries marked `% TODO verify` need checking.

## Files
- `main.tex`: article skeleton covering the introduction, literature, model, solution method, verification, empirical targets, planned results, extensions and conclusion.
- `refs.bib`: the bibliography. Uncertain entries are marked `% TODO verify`.

## Compile
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
This needs a standard TeX Live installation with amsmath, natbib, hyperref and url.

## Dependencies
- `dl-jmp/model/`: model code and calibration. The Model section must stay consistent with it.
- `dl-jmp/toolkit/`: the deep-equilibrium-net solver and the linear verification tools behind the Solution Method and Verification sections.
- `dl-jmp/literature/`: EconResearcher's review. The Related Literature section and `refs.bib` will be filled and verified from it.
