# MACE4D figures

Annotated figures of `MACE4DModel`: the scan divided into frames, one frame's
data-fit agent, the three hyperplane denoisers, the consensus loop, where each
parameter comes from, and where the work runs.  They are drafts for the deck
and, later, for the Sphinx page.

## Files

- `notation.tex`: every parameter, symbol, default, and constant the figures
  show, as macros.  `\codenamestrue` renders code names (the current setting);
  `\codenamesfalse` renders the symbolic form.  Change a name, a symbol, or a
  default here and every figure follows.
- `styles.tex`: colors and TikZ styles.  Blue is the data-fit side, green the
  priors, amber the consensus state.  A solid arrow is a value the user set, a
  dashed arrow an estimate.
- `figpreamble.tex`: the shared preamble of the standalone figures.
- `fig1_frames.tex` to `fig6_devices.tex`: one standalone figure each.
- `build.sh`: renders each figure to `build/` as PDF (deck), SVG (docs), and
  PNG (a quick look).  `bash build.sh` builds all; name a file to build one.
- `mace4d_figures_deck.tex`: a beamer deck with one figure per slide.  Build
  with `latexmk mace4d_figures_deck.tex` after `bash build.sh`.

`build/` is ignored by git.  Build the figures before the deck.

## Notation

The figures use code names now, with a glossary line tying each code name to
its symbol where a formula needs one.  To switch the figures to symbols, set
`\codenamesfalse` in `notation.tex` and rebuild.  A symbol or default lives in
exactly one macro, so a change is one edit.

## Where the figures go

The deck uses the PDFs from `build/`.  The Sphinx page will use the SVGs,
committed into the mbirtorch docs so that the docs build needs no LaTeX.
