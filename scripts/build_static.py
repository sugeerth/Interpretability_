"""Bundle web/ into dist/: a static site (GitHub Pages) plus dist/artifact.html,
the same page without the document skeleton for hosts that add their own."""
import re
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
dist = root / "dist"
shutil.rmtree(dist, ignore_errors=True)
shutil.copytree(root / "web", dist)
html = (dist / "index.html").read_text()
body = re.sub(r"<!doctype html>\s*<meta charset[^>]*>\s*<meta name=\"viewport\"[^>]*>\s*", "", html, count=1)
assert body != html, "skeleton header not found"
(dist / "artifact.html").write_text(body)
print("built", dist, sum(f.stat().st_size for f in dist.rglob("*") if f.is_file()) // 1024, "KB")
