#!/usr/bin/env python
"""Compile the manuscript's vector protocol schematic from its TikZ source.

This is a conceptual diagram, not a rendering of experimental outcomes.
Requires pdflatex (TeX Live) and pdftocairo (Poppler); no shell escape,
download, model execution, or statistical analysis is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "paper/manuscript/figures/hidden_order_protocol.tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "paper/manuscript/generated/hidden_order_protocol.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.suffix.lower() != ".pdf":
        raise ValueError("--output must name a PDF file")
    for executable in ("pdflatex", "pdftocairo"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required rendering executable is unavailable: {executable}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    with tempfile.TemporaryDirectory(prefix="hidden-order-protocol-") as temporary:
        build = Path(temporary)
        subprocess.run(
            [
                "pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
                "-halt-on-error", f"-output-directory={build}", str(SOURCE),
            ],
            cwd=SOURCE.parent,
            check=True,
        )
        pdf = build / SOURCE.with_suffix(".pdf").name
        svg = build / SOURCE.with_suffix(".svg").name
        subprocess.run(
            ["pdftocairo", "-svg", str(pdf), str(svg)],
            check=True,
        )
        # Publish the pair only after both renderers have succeeded.
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, output)
        shutil.copyfile(svg, output.with_suffix(".svg"))
    print(output)


if __name__ == "__main__":
    main()
