"""Combine the OAuth screenshots and the upload evidence into one PDF for the audit form.

    python scripts/build_audit_bundle.py consent.png scopes.png revocation.png

The form has a single "conditional evidence" field for all of it, and accepts one file, so
the parts have to arrive as one document. Order, which is the order the form lists them in:
consent screen, scopes, revocation, then the upload evidence.

Any number of screenshots may be given; they are placed in the order written on the command
line, each on its own page with the caption taken from --captions if supplied. The upload
evidence (docs/upload-evidence.pdf, built by build_audit_evidence.py) is appended last.

Images are laid out by a browser and printed, rather than drawn into a PDF directly, so a
wide screenshot is fitted to the page the way it would look on screen instead of being
cropped or stretched.
"""
import argparse
import base64
import html
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EVIDENCE = DOCS / "upload-evidence.pdf"
OUT = DOCS / "oauth-and-upload-evidence.pdf"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
DEFAULT_CAPTIONS = [
    "OAuth consent screen, as the channel owner sees it when authorising the application",
    "The permission requested: one scope, youtube.upload",
    "Revoking the application's access, in the owner's own Google account settings",
]


def shots_in_docs():
    """Screenshots dropped into docs/, in filename order.

    So the bundle can be built with no arguments at all: name the files 1-consent.png,
    2-scopes.png, 3-revocation.png and run the script. Sorting is by name, not by the time
    the files were made, because the order the form wants is not the order they were taken.
    """
    found = sorted(p for p in DOCS.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return [str(p) for p in found]


def find_browser():
    for path in BROWSERS:
        if Path(path).exists():
            return path
    raise SystemExit("No Edge or Chrome found to render the pages; install one or print by hand.")


def images_pdf(images, captions, out_pdf):
    """One page per image, printed by a browser so each is fitted rather than cropped."""
    blocks = []
    for i, image in enumerate(images):
        data = base64.b64encode(Path(image).read_bytes()).decode()
        suffix = Path(image).suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
        caption = captions[i] if i < len(captions) else Path(image).name
        blocks.append(
            f'<section><h2>{i + 1}. {html.escape(caption)}</h2>'
            f'<img src="data:image/{mime};base64,{data}" alt="">'
            f'<p class="src">{html.escape(Path(image).name)}</p></section>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>yt-factory - OAuth flow</title><style>
  body {{ font: 15px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif; color:#111;
         margin: 1.4rem; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .2rem; }}
  .sub {{ color:#555; margin:0 0 1.2rem; font-size: 13.5px; }}
  section {{ page-break-after: always; break-after: page; }}
  section:last-of-type {{ page-break-after: auto; break-after: auto; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 .5rem; }}
  img {{ max-width: 100%; max-height: 8.4in; border: 1px solid #ccc; display: block; }}
  .src {{ color:#777; font-size: 11.5px; margin-top: .35rem; }}
</style></head><body>
<h1>yt-factory — OAuth flow</h1>
<p class="sub">Project yt-factory-510411 &middot; scope
<code>https://www.googleapis.com/auth/youtube.upload</code> &middot; the application has one
user, the owner of the channel it publishes to. Evidence of the upload flow follows these
screenshots.</p>
{"".join(blocks)}
</body></html>"""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "oauth.html"
        source.write_text(page, encoding="utf-8")
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--virtual-time-budget=15000", "--print-to-pdf-no-header",
                        f"--print-to-pdf={out_pdf}", source.as_uri()],
                       capture_output=True, timeout=180)
    if not Path(out_pdf).exists():
        raise SystemExit(f"the browser did not produce {out_pdf}")
    return out_pdf


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="screenshots, in the order they should appear; "
                                                 "default: the images in docs/, in filename order")
    parser.add_argument("--captions", nargs="*", default=DEFAULT_CAPTIONS)
    parser.add_argument("--evidence", default=str(EVIDENCE))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    images = list(args.images) or shots_in_docs()
    if not images:
        raise SystemExit(f"no screenshots given and none in {DOCS}; put the three there "
                         "(1-consent.png, 2-scopes.png, 3-revocation.png) or name them "
                         "on the command line")
    missing = [p for p in images + [args.evidence] if not Path(p).exists()]
    if missing:
        raise SystemExit("not found: " + ", ".join(missing))
    args.images = images

    from pypdf import PdfWriter
    with tempfile.TemporaryDirectory() as tmp:
        shots = images_pdf(args.images, args.captions, str(Path(tmp) / "shots.pdf"))
        writer = PdfWriter()
        for part in (shots, args.evidence):
            writer.append(part)
        with open(args.out, "wb") as f:
            writer.write(f)

    from pypdf import PdfReader
    pages = len(PdfReader(args.out).pages)
    size = Path(args.out).stat().st_size
    print(f"{args.out}\n{pages} pages, {size / 1024:.0f} KB "
          f"({len(args.images)} screenshot(s) then the upload evidence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
