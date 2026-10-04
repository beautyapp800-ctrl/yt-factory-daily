"""Build docs/upload-evidence.html: the log of a real upload plus the code that performed it.

    python scripts/build_audit_evidence.py

Then print it, and the three published pages, to PDF with headless Edge or Chrome:

    msedge --headless=new --print-to-pdf-no-header\\
        --print-to-pdf=C:/yt-factory/docs/upload-evidence.pdf\\
        file:///C:/yt-factory/docs/upload-evidence.html


The audit form expects screenshots or a demo video of the application. This one has no user
interface to film - it runs unattended on a server - so the equivalent evidence is the run log
of a real upload and the source that produced it, side by side.
"""
import html
import re
import subprocess
from pathlib import Path

ROOT = Path(r"C:\yt-factory")
RUN = "37185441013"
REPO = "beautyapp800-ctrl/yt-factory-daily"


def run_log_lines():
    """The upload stage's own lines from the finished GitHub Actions run."""
    out = subprocess.run(["gh", "run", "view", RUN, "--repo", REPO, "--log"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    lines = []
    for line in out.splitlines():
        if "| upload " in line or "| seo " in line or "| thumbnail " in line:
            # strip the workflow's step/timestamp prefix, keep the application's own line
            lines.append(re.sub(r"^.*?\dZ\s*", "", line).rstrip())
    return lines


def excerpt(path, start_marker, end_marker, keep=None):
    """Lines of a source file between two markers, with the original line numbers."""
    text = (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    start = next(i for i, l in enumerate(text) if start_marker in l)
    end = next(i for i in range(start + 1, len(text)) if end_marker in text[i])
    rows = [(i + 1, text[i]) for i in range(start, end + 1)]
    if keep:
        rows = [(n, l) for n, l in rows if any(k in l for k in keep) or not l.strip()
                or l.strip().startswith("#") or l.strip().startswith('"')]
    return rows


def code_block(path, rows):
    body = "\n".join(
        f'<span class="ln">{(str(n) if n else ""):>4}</span> {html.escape(line)}'
        for n, line in rows)
    return f'<div class="file">{html.escape(path)}</div><pre class="code">{body}</pre>'


log = run_log_lines()
insert_rows = excerpt("pipeline/upload.py", "def build_body(video, cfg, now=None):",
                      'return {')
insert_rows += [(None, "        ... (the request body returned here is sent to videos.insert)")]
send_rows = excerpt("pipeline/upload.py", "    request = service.videos().insert(",
                    "stats = {")
flags_rows = excerpt("pipeline/upload.py", "# Set on every upload, not read from config",
                     "CONTAINS_SYNTHETIC_MEDIA = True")
assert_rows = excerpt("pipeline/upload.py", "def assert_mandatory_flags(body):",
                      '"YouTube requires AI-generated content to be declared")')
publish_rows = excerpt("pipeline/upload.py", "def publish_at(cfg, now=None):",
                       'return when.astimezone(timezone.utc)')

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>yt-factory - evidence of the upload flow</title>
<style>
  body {{ font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; color: #111;
          max-width: 46rem; margin: 2rem auto; padding: 0 1.2rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #ddd;
        padding-bottom: .25rem; }}
  .sub {{ color: #555; margin-top: 0; }}
  p {{ margin: .7rem 0; }}
  pre {{ background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 5px;
         padding: .7rem .8rem; font: 11.5px/1.45 "Cascadia Mono", Consolas, monospace;
         white-space: pre-wrap; word-break: break-word; }}
  .code .ln {{ color: #9aa0a6; user-select: none; }}
  .file {{ font: 11.5px "Cascadia Mono", Consolas, monospace; color: #444;
           background: #eceff1; border: 1px solid #e1e4e8; border-bottom: none;
           border-radius: 5px 5px 0 0; padding: .3rem .6rem; }}
  .file + pre {{ border-radius: 0 0 5px 5px; margin-top: 0; }}
  .note {{ color: #444; font-size: 13.5px; }}
</style></head><body>

<h1>yt-factory — evidence of the upload flow</h1>
<p class="sub">Project <b>yt-factory-510411</b> &middot; OAuth client
147757322186-hj1csktsf05hq0jj4g43frlhmis9r1sr.apps.googleusercontent.com &middot;
scope <code>youtube.upload</code></p>

<p class="note">This application has no user interface to screenshot: it runs unattended on a
server and publishes to one channel belonging to its owner. The equivalent evidence is below —
the log of a real upload, and the source code that performed it. The run can be inspected in
full at
<a href="https://github.com/{REPO}/actions/runs/{RUN}">github.com/{REPO}/actions/runs/{RUN}</a>,
and the source at <a href="https://github.com/{REPO}">github.com/{REPO}</a>.</p>

<h2>1. Log of a real upload</h2>
<p>GitHub Actions run {RUN}, 4 October 2026. The video was uploaded private with
<code>publishAt</code> set three days ahead, and its thumbnail set in a second call. The video
is <a href="https://youtu.be/gRePZohxtuM">youtu.be/gRePZohxtuM</a>, on the owner's own channel.</p>
<pre>{html.escape(chr(10).join(log))}</pre>

<h2>2. The code that calls videos.insert with publishAt</h2>
<p>The request body. When a video is scheduled it is uploaded <b>private</b> and YouTube is
asked to publish it at a stated moment, which is what <code>status.publishAt</code> means.</p>
{code_block("pipeline/upload.py", insert_rows)}

<p>The moment itself is computed through the real time zone, so 21:00 local is correct in both
summer and winter:</p>
{code_block("pipeline/upload.py", publish_rows)}

<p>And the call, a resumable upload of the rendered file:</p>
{code_block("pipeline/upload.py", send_rows)}

<h2>3. The code that sets containsSyntheticMedia</h2>
<p>The narration is synthetic speech and the artwork is generated, so every upload declares it.
These are constants, not configuration: no setting can switch them off.</p>
{code_block("pipeline/upload.py", flags_rows)}

<p>They are checked on the request itself immediately before it is sent, and the upload is
refused rather than sent undeclared:</p>
{code_block("pipeline/upload.py", assert_rows)}

</body></html>
"""
out = ROOT / "docs" / "upload-evidence.html"
out.write_text(page, encoding="utf-8")
print("wrote", out, len(page), "bytes;", len(log), "log lines")
