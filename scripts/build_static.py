"""Bundle web/ into dist/: a static site plus dist/artifact.html, the same page
without the document skeleton for hosts that add their own.

    python scripts/build_static.py                       # GitHub Pages / any static host
    python scripts/build_static.py --site URL [--out D]  # a page on an existing site: adds lang, description,
                                                         # canonical and social-card tags for that URL
"""
import argparse
import re
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--site", help="public URL the page will live at, ending in /")
ap.add_argument("--out", help="output directory (default: dist/)")
ap.add_argument("--card", default="card.png", help="social-card image file name placed next to the page (1200x630)")
ap.add_argument("--favicon", default="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
                                      "<path d='M50 12 90 84H10z' fill='%234a3fd0'/></svg>")
a = ap.parse_args()

root = Path(__file__).resolve().parents[1]
dist = Path(a.out) if a.out else root / "dist"
shutil.rmtree(dist, ignore_errors=True)
shutil.copytree(root / "web", dist)
html = (dist / "index.html").read_text()
body = re.sub(r"<!doctype html>\s*<meta charset[^>]*>\s*<meta name=\"viewport\"[^>]*>\s*", "", html, count=1)
assert body != html, "skeleton header not found"
for f in (dist / "bundles").glob("*.json"):       # tokens that are partial UTF-8 sequences decode to U+FFFD
    f.write_text(f.read_text().replace("�", "□"))
if a.site:
    title = "Prism: see inside a language model"
    desc = ("Interactive interpretability explorer: follow one prediction through Gemma 3, GPT-2, Qwen and Pythia, "
            "from input tokens through attention heads to the output.")
    assert 70 <= len(desc) <= 165, len(desc)
    head = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{desc}">
<link rel="canonical" href="{a.site}">
<link rel="icon" href="{a.favicon}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:type" content="website">
<meta property="og:url" content="{a.site}">
<meta property="og:image" content="{a.site}{a.card}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:site_name" content="Sugeerth Murugesan">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:image" content="{a.site}{a.card}">
'''
    page = re.sub(r"<!doctype html>\s*<meta charset[^>]*>\s*<meta name=\"viewport\"[^>]*>\s*<meta name=\"claude:[^>]*>\s*", head, html, count=1)
    assert page != html
    page = page.replace("<title>Prism Interpretability</title>", f"<title>{title}</title>", 1)
    page = page.replace("</style>\n", "</style>\n</head>\n<body>\n", 1) + "\n</body>\n</html>\n"
    page = page.replace("<script>\n\"use strict\";", "<script>window.PRISM_STATIC = true;</script>\n<script>\n\"use strict\";", 1)   # no API on a static host
    assert "PRISM_STATIC = true" in page
    (dist / "index.html").write_text(page)
else:
    (dist / "artifact.html").write_text(body)
print("built", dist, sum(f.stat().st_size for f in dist.rglob("*") if f.is_file()) // 1024, "KB")
