"""Render PAPER.md -> PAPER.pdf via markdown -> styled HTML -> headless Chrome.

No LaTeX toolchain needed. Uses python-markdown (tables + fenced code) for the
HTML, an academic single-column print stylesheet, and Google Chrome's
--headless --print-to-pdf for rendering (best table/typography fidelity of the
tools available on this machine).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 22mm 20mm; }
html { -webkit-print-color-adjust: exact; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.5; color: #111;
  max-width: 100%; margin: 0;
}
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 .4em; font-weight: 700; }
h2 { font-size: 13.5pt; margin: 1.4em 0 .4em; border-bottom: 1px solid #ccc;
     padding-bottom: .15em; }
h3 { font-size: 11.5pt; margin: 1.1em 0 .3em; }
h1 + p, h1 + p + p, h1 + p + p + p { text-align: left; }
p { margin: 0 0 .6em; text-align: justify; }
strong { font-weight: 700; }
em { font-style: italic; }
code, pre {
  font-family: "SF Mono", "Menlo", "Consolas", monospace; font-size: 9pt;
}
pre { background: #f6f6f6; padding: .7em .9em; border-radius: 4px;
      overflow-x: auto; white-space: pre-wrap; }
code { background: #f0f0f0; padding: .05em .3em; border-radius: 3px; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 1.1em auto 0.3em; }
img + em, p > img { page-break-inside: avoid; }
table { border-collapse: collapse; margin: 1em 0; font-size: 9.2pt; width: auto; }
th, td { border: 1px solid #bbb; padding: .3em .6em; text-align: left; }
th { background: #efefef; font-weight: 700; }
blockquote { margin: .6em 0; padding-left: 1em; border-left: 3px solid #ddd;
             color: #444; }
a { color: #1a4f8b; text-decoration: none; }
h2, h3 { page-break-after: avoid; }
table, pre, figure { page-break-inside: avoid; }
"""


def main(src="paper/PAPER.md", out="paper/PAPER.pdf"):
    root = pathlib.Path(__file__).resolve().parent.parent
    src_p, out_p = root / src, root / out
    text = src_p.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    # HTML lives beside the markdown so relative figure paths resolve
    html_p = out_p.parent / (out_p.stem + ".html")
    html_p.write_text(html, encoding="utf-8")

    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={out_p}", html_p.as_uri()],
        check=True, capture_output=True,
    )
    print(f"wrote {out_p}  ({out_p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(*sys.argv[1:])
