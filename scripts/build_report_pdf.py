#!/usr/bin/env python3
"""Renders docs/report.md to docs/report.pdf (US Letter) with headless Google Chrome.

    python3 scripts/build_report_pdf.py [--chrome PATH] [--keep-html]

Only the Markdown subset used by the report is supported: headings, paragraphs, bullet and
numbered lists, pipe tables, images, fenced code, inline code/bold/italic/links, and raw <div>
lines (`<div class="pagebreak"></div>` starts a new page). Prints the page count and the page on
which the appendix starts.
"""
import argparse
import html
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 0.6in 0.65in 0.6in 0.65in; }
body { font-family: "Times New Roman", Times, serif; font-size: 10.5pt; line-height: 1.22;
       color: #111; margin: 0; }
h1 { font-size: 15.5pt; margin: 0 0 3pt 0; }
h2 { font-size: 12pt; margin: 9pt 0 3pt 0; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; }
h3 { font-size: 10.8pt; margin: 7pt 0 2pt 0; }
p { margin: 3pt 0; text-align: left; }
ul, ol { margin: 2pt 0 3pt 0; padding-left: 16pt; }
li { margin: 1pt 0; text-align: left; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.6pt; }
pre { font-size: 8pt; background: #f4f4f2; padding: 4pt 6pt; margin: 3pt 0; white-space: pre-wrap; }
pre code { font-size: 8pt; }
table { border-collapse: collapse; margin: 4pt auto 5pt auto; font-size: 8.4pt; line-height: 1.15;
        page-break-inside: avoid; }
th, td { border: 0.5pt solid #aaa; padding: 1.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #eeede9; }
figure { margin: 4pt 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figure.wide img { width: 100%; }
.pagebreak { page-break-before: always; }
h1, h2, h3 { page-break-after: avoid; break-after: avoid; }
a { color: #1c4f9c; text-decoration: none; }
"""

ALLOWED_TAG = re.compile(r"<(?=/?(sub|sup|br|div|span)\b)")


def inline(text):
    # Protect code spans first.
    codes = []

    def keep_code(m):
        codes.append(html.escape(m.group(1)))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", keep_code, text)
    text = text.replace("&", "&amp;")
    text = re.sub(r"<(?!/?(sub|sup|br|div|span)\b)", "&lt;", text)
    text = text.replace("\\|", "|")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", text)
    return text


def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells = re.split(r"(?<!\\)\|", line)
    return [c.strip() for c in cells]


def convert(md):
    lines = md.splitlines()
    out = []
    i = 0
    para = []

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_para()
            i += 1
            continue
        if stripped.startswith("```"):
            flush_para()
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            flush_para()
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if stripped.startswith("<div"):
            flush_para()
            out.append(stripped)
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m:
            flush_para()
            out.append(f'<figure class="wide"><img src="{m.group(2)}" alt="{html.escape(m.group(1))}"></figure>')
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-{3,}", lines[i + 1].strip()):
            flush_para()
            header = split_row(stripped)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            t = ["<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead><tbody>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue
        m = re.match(r"^([*-]|\d+\.)\s+(.*)$", stripped)
        if m and not line.startswith("  "):
            flush_para()
            ordered = m.group(1)[0].isdigit()
            items = []
            while i < len(lines):
                cur = lines[i]
                mm = re.match(r"^([*-]|\d+\.)\s+(.*)$", cur.strip())
                if mm and not cur.startswith("  ") and (mm.group(1)[0].isdigit() == ordered):
                    items.append([mm.group(2)])
                elif cur.startswith("  ") and cur.strip() and items:
                    items[-1].append(cur.strip())
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join("<li>" + inline(" ".join(it)) + "</li>" for it in items) + f"</{tag}>")
            continue
        para.append(stripped)
        i += 1
    flush_para()
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default=DEFAULT_CHROME)
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()
    src = os.path.join(DOCS, "report.md")
    tmp_html = os.path.join(DOCS, ".report_build.html")
    pdf = os.path.join(DOCS, "report.pdf")
    with open(src, encoding="utf-8") as f:
        body = convert(f.read())
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>HNSW report</title>"
                f"<style>{CSS}</style></head><body>{body}</body></html>")
    if not os.path.exists(args.chrome):
        sys.exit(f"Chrome not found at {args.chrome}; open {tmp_html} in a browser and print to PDF")
    cmd = [args.chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
           f"--print-to-pdf={pdf}", "file://" + tmp_html]
    subprocess.run(cmd, check=True, capture_output=True)
    if not args.keep_html:
        os.remove(tmp_html)
    try:
        import fitz  # PyMuPDF, optional: page statistics only
        doc = fitz.open(pdf)
        appendix = next((p.number + 1 for p in doc if "Appendix A. Additional results" in p.get_text()), None)
        print(f"wrote {pdf}: {doc.page_count} pages; appendix starts on page {appendix} "
              f"(main body = {appendix - 1 if appendix else doc.page_count} pages)")
    except ImportError:
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
