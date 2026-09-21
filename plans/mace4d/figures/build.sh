#!/usr/bin/env bash
# Build the figures named on the command line, or all of them, into build/:
# a PDF for the deck, an SVG for the docs, and a PNG for a quick look.
#   bash build.sh                 # every fig<n>_*.tex
#   bash build.sh fig1_frames.tex # one figure
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
figs=("$@")
if [ ${#figs[@]} -eq 0 ]; then figs=(fig[0-9]*.tex); fi
for f in "${figs[@]}"; do
  name="${f%.tex}"
  if ! pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$f" > "build/$name.stdout" 2>&1; then
    echo "FAILED: $f"
    grep -n -A4 '^!' "build/$name.log" | head -40
    exit 1
  fi
  pdftocairo -svg "build/$name.pdf" "build/$name.svg"
  pdftocairo -png -r 110 -singlefile "build/$name.pdf" "build/$name"
  echo "built $name -> build/$name.{pdf,svg,png}"
done
