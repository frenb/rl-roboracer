# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai>=1.20"]
# ///
"""
Real-to-sim, step 1: a walk-around video of the physical track -> a measured
top-down layout (meters) and a texture of the mat's markings, by Gemini.

    uv run scripts/real2sim/video_to_layout.py unity/Videos/IMG_5748.MOV --name home-track

Writes unity/Assets/Real2Sim/<name>/:
    layout.json   mat, road centerline, islands, edge lines, crosswalks,
                  arrows, fence panels, start pose (read by Real2SimCourse.cs
                  and build_blender_model.py)
    mat.svg       the mat seen from above, 1 SVG unit = 1 mm
    mat.png       mat.svg rendered by headless Edge or Chrome (the road texture)
    top-view.*    a copy of --top-view, when given

Layout frame: origin at one corner of the mat, +x along its width, +y along
its length, seen from above; headings in degrees clockwise from +y. Unity
maps x -> X and y -> Z, so the headings are Unity yaw.

Scale comes from what Gemini can see (fence panels are a uniform size, the
mat is a standard print). Pass a known measurement with --panel-width or
--mat-size to pin it. The API key is GOOGLE_GEMINI_API_KEY, GEMINI_API_KEY
or GOOGLE_API_KEY, from the environment or the repo's .env.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from google import genai
from google.genai import types

REPO = Path(__file__).resolve().parents[2]
KEY_VARS = ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")

LAYOUT_PROMPT = """\
This video walks around a small indoor track for a 1/10-scale autonomous RC car
(a JetRacer, wheelbase about 0.24 m). The track is a printed road mat on the
floor, enclosed and divided by modular fence panels. Build a measured
top-down map of it for a simulator.

Watch the whole video, relate every view to the others, and return ONE JSON
object, no prose, with exactly this structure (all lengths in meters):

{
  "units": "meters",
  "scale_reference": "<what you used to set scale and how>",
  "confidence": "<what is certain, what is estimated>",
  "mat": {"width": <x extent>, "length": <y extent>, "base_color": "#rrggbb"},
  "road": {"width": <drivable lane width between the edge lines>, "closed": <true if the road is a loop>,
           "centerline": [{"x": 0, "y": 0}, ...]},
  "islands": [{"label": "<e.g. center block>", "color": "#rrggbb", "corner_radius": <m>,
               "points": [{"x": 0, "y": 0}, ...]}],
  "edge_lines": [{"color": "#rrggbb", "width": <m>, "closed": <bool>, "points": [{"x": 0, "y": 0}, ...]}],
  "crosswalks": [{"x": 0, "y": 0, "heading_deg": 0, "length": <across the road>, "depth": <along the road>, "stripes": <count>}],
  "arrows": [{"x": 0, "y": 0, "heading_deg": 0, "kind": "straight|left|right|straight_left|straight_right|u_turn", "length": <m>}],
  "fence_panels": [{"x1": 0, "y1": 0, "x2": 0, "y2": 0, "height": <m>}],
  "start": {"x": 0, "y": 0, "heading_deg": 0},
  "notes": "<anything a simulator builder should know>"
}

Frame: origin at one corner of the mat; +x along the mat's width, +y along
its length, as seen from above. heading_deg is clockwise from +y (0 = +y,
90 = +x). Rules:
- centerline: the middle of the drivable lane, in driving order, dense enough
  that straight segments between points stay on the road (a point at least
  every 0.25 m around curves). If the road is a loop, do not repeat the first
  point at the end.
- islands: the non-road areas inside the mat (the green blocks), as polygons
  of their outer edge in order; corner_radius for their rounded corners.
- edge_lines: the painted yellow lines, as polylines along their middle.
- fence_panels: ONE entry per physical panel, from post to post, including the
  panels around the outside and any that sit on islands. Count them; panels
  are a uniform width, so use that to check the map's scale.
- start: a sensible spot on the centerline, heading in the driving direction.
"""

SVG_PROMPT = """\
Here is the measured top-down layout of the track mat in this video, as JSON:

{layout}

Draw the mat as it looks from directly above, as one SVG document, so it can
be used as the road texture in a simulator. Requirements:
- <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w_mm} {l_mm}" width="{w_mm}" height="{l_mm}">;
  1 SVG unit = 1 mm. The layout's (x, y) in meters maps to SVG
  (x * 1000, ({length} - y) * 1000): SVG y points down, so the layout's +y
  is up in the image.
- Fill the whole mat with its base color, then draw what is printed on it,
  matching the video: green islands with rounded corners, the yellow edge
  lines at their real width, white crosswalk stripes, white turn arrows, and
  any other printed markings you can see. Use the layout's positions; use the
  video for the look (colors, stripe counts, arrow shapes).
- Do NOT draw the fence panels, the floor around the mat, shadows, folds or
  wrinkles: only the printed design.
- Plain shapes only (rect, path, polygon, polyline, circle, line, g). No text,
  images, filters, gradients, CSS or scripts.
Return only the SVG, starting with <svg and ending with </svg>.
"""


def api_key() -> str:
    for v in KEY_VARS:
        if os.environ.get(v):
            return os.environ[v]
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(\w+)\s*=\s*(.*)$", line)
            if m and m.group(1) in KEY_VARS and m.group(2).strip().strip("\"'"):
                return m.group(2).strip().strip("\"'")
    sys.exit(f"No Gemini API key: set one of {', '.join(KEY_VARS)} or add it to {env}")


def upload(client: genai.Client, video: Path, out: Path) -> types.File:
    """Upload once; reuse the uploaded file while Google still has it (48 h)."""
    cache = out / "upload.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("size") == video.stat().st_size:
            try:
                f = client.files.get(name=c["name"])
                if f.state and f.state.name == "ACTIVE":
                    print(f"reusing upload {f.name}")
                    return f
            except Exception:
                pass
    print(f"uploading {video.name} ({video.stat().st_size / 1e6:.0f} MB)")
    f = client.files.upload(file=str(video), config={"mime_type": "video/quicktime" if video.suffix.lower() == ".mov" else "video/mp4"})
    while f.state and f.state.name == "PROCESSING":
        time.sleep(5)
        f = client.files.get(name=f.name)
    if not f.state or f.state.name != "ACTIVE":
        sys.exit(f"upload failed: {f.state}")
    cache.write_text(json.dumps({"name": f.name, "size": video.stat().st_size}))
    return f


TOP_VIEW_NOTE = """
The attached still image is a clean top-down view of the same mat. It is the
authoritative source for everything printed on the mat (islands and their
shapes, yellow lines, crosswalks, arrows, stickers) and for its proportions;
use the video for the fence panels and anything the image does not show.
"""


def ask(client, model, video_file, fps, prompt, json_out: bool, thinking: int, images=()) -> str:
    video = types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type=video_file.mime_type),
                       video_metadata=types.VideoMetadata(fps=fps))
    stills = [types.Part.from_bytes(data=p.read_bytes(), mime_type="image/png" if p.suffix.lower() == ".png" else "image/jpeg")
              for p in images]
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json" if json_out else "text/plain",
        max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking) if thinking else None)
    for attempt in range(3):
        try:
            r = client.models.generate_content(model=model, contents=[video, *stills, prompt + (TOP_VIEW_NOTE if stills else "")], config=cfg)
            if r.text:
                return r.text
            print(f"empty response ({r.candidates[0].finish_reason if r.candidates else 'no candidates'}); retrying")
        except Exception as e:
            if attempt == 2:
                raise
            print(f"request failed ({e}); retrying")
        time.sleep(10)
    sys.exit("Gemini returned no content")


def check_layout(L: dict) -> list[str]:
    problems = []
    for k in ("mat", "road", "fence_panels", "start"):
        if k not in L:
            problems.append(f"missing '{k}'")
    if problems:
        return problems
    w, l = L["mat"].get("width", 0), L["mat"].get("length", 0)
    if not (0.3 < w < 20 and 0.3 < l < 20):
        problems.append(f"mat size {w} x {l} m is implausible")
    if len(L["road"].get("centerline", [])) < 3:
        problems.append("road centerline has fewer than 3 points")
    if not L["fence_panels"]:
        problems.append("no fence panels")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--name", help="output folder under unity/Assets/Real2Sim (default: the video's name)")
    ap.add_argument("--model", default="gemini-4-argon")
    ap.add_argument("--fps", type=float, default=2, help="video frames per second Gemini samples")
    ap.add_argument("--top-view", type=Path, help="a clean top-down image of the mat (authoritative for the printed layout)")
    ap.add_argument("--note", default="", help="extra guidance for Gemini, e.g. an artifact to ignore in the top view")
    ap.add_argument("--panel-width", type=float, help="known width of one fence panel, meters")
    ap.add_argument("--panel-height", type=float, help="known height of one fence panel, meters")
    ap.add_argument("--mat-size", help="known mat size, WIDTHxLENGTH in meters (x along the width, y along the length), e.g. 3.2x2.8")
    ap.add_argument("--thinking", type=int, default=0, help="thinking token budget (0 = model default)")
    ap.add_argument("--texture-px", type=int, default=2048, help="long side of mat.png")
    ap.add_argument("--skip-layout", action="store_true", help="keep the existing layout.json, redo only the texture")
    ap.add_argument("--rasterize-only", action="store_true", help="only render the existing mat.svg to mat.png")
    a = ap.parse_args()

    video = a.video if a.video.is_absolute() else (Path.cwd() / a.video)
    if not video.exists():
        sys.exit(f"no such video: {video}")
    name = a.name or re.sub(r"[^a-z0-9]+", "-", video.stem.lower()).strip("-")
    out = REPO / "unity" / "Assets" / "Real2Sim" / name
    out.mkdir(parents=True, exist_ok=True)
    if a.rasterize_only:
        rasterize(out / "mat.svg", out / "mat.png")
        return

    images = []
    if a.top_view:
        if not a.top_view.exists():
            sys.exit(f"no such image: {a.top_view}")
        images = [a.top_view]
        kept = out / ("top-view" + a.top_view.suffix.lower())
        if a.top_view.resolve() != kept.resolve():
            kept.write_bytes(a.top_view.read_bytes())
    extra = ""
    known = []
    if a.panel_width:
        known.append(f"Each fence panel is {a.panel_width} m wide (post to post).")
    if a.panel_height:
        known.append(f"Each fence panel is {a.panel_height} m tall.")
    mat_size = None
    if a.mat_size:
        mat_size = tuple(float(v) for v in a.mat_size.lower().split("x"))
        known.append(f"The mat is exactly {mat_size[0]} m along x (its width) by {mat_size[1]} m along y (its length).")
    if known:
        extra += "\nKnown measurements (exact; use them): " + " ".join(known) + "\n"
    if a.note:
        extra += f"\nNote: {a.note}\n"

    client = genai.Client(api_key=api_key())
    vf = upload(client, video, out)

    layout_file = out / "layout.json"
    if a.skip_layout and layout_file.exists():
        layout = json.loads(layout_file.read_text(encoding="utf-8"))
    else:
        print(f"asking {a.model} for the layout")
        t0 = time.time()
        raw = ask(client, a.model, vf, a.fps, LAYOUT_PROMPT + extra, True, a.thinking, images)
        (out / "layout.raw.txt").write_text(raw, encoding="utf-8")
        try:
            layout = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
        except json.JSONDecodeError as e:
            sys.exit(f"the layout is not valid JSON ({e}); see {out / 'layout.raw.txt'}")
        problems = check_layout(layout)
        if not problems and mat_size:
            gw, gl = float(layout["mat"]["width"]), float(layout["mat"]["length"])
            if abs(gw - mat_size[0]) > 0.05 * mat_size[0] or abs(gl - mat_size[1]) > 0.05 * mat_size[1]:
                problems.append(f"mat came back {gw} x {gl} m, not the given {mat_size[0]} x {mat_size[1]} m")
        if problems:
            sys.exit("layout rejected: " + "; ".join(problems) + f". See {out / 'layout.raw.txt'}")
        layout = {"name": name, "source_video": video.name, "model": a.model, **layout}
        layout_file.write_text(json.dumps(layout, indent=2), encoding="utf-8")
        print(f"layout: mat {layout['mat']['width']} x {layout['mat']['length']} m, road {layout['road'].get('width')} m wide, "
              f"{len(layout['road']['centerline'])} centerline points, {len(layout['fence_panels'])} fence panels "
              f"({time.time() - t0:.0f} s)")

    w, l = float(layout["mat"]["width"]), float(layout["mat"]["length"])
    print(f"asking {a.model} for the mat texture")
    svg_prompt = SVG_PROMPT.format(layout=json.dumps({k: v for k, v in layout.items() if k != "fence_panels"}, indent=1),
                                   w_mm=round(w * 1000), l_mm=round(l * 1000), length=l)
    raw = ask(client, a.model, vf, a.fps, svg_prompt + extra, False, a.thinking, images)
    m = re.search(r"<svg\b.*</svg>", raw, re.S)
    if not m:
        (out / "mat.raw.txt").write_text(raw, encoding="utf-8")
        sys.exit(f"no SVG in the response; see {out / 'mat.raw.txt'}")
    svg = m.group(0)
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        (out / "mat.raw.txt").write_text(raw, encoding="utf-8")
        sys.exit(f"the SVG does not parse ({e}); see {out / 'mat.raw.txt'}")
    # Render size comes from width/height; the viewBox keeps the mm drawing.
    scale = a.texture_px / max(w, l)
    root.set("viewBox", f"0 0 {round(w * 1000)} {round(l * 1000)}")
    root.set("width", str(round(w * scale)))
    root.set("height", str(round(l * scale)))
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    (out / "mat.svg").write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")

    rasterize(out / "mat.svg", out / "mat.png")
    print(f"done: {out}")


def rasterize(svg: Path, png: Path):
    """mat.svg -> mat.png at the SVG's width/height, with headless Edge or Chrome."""
    root = ET.parse(svg).getroot()
    wpx, hpx = int(float(root.get("width"))), int(float(root.get("height")))
    browsers = [Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe",
                Path(os.environ.get("ProgramFiles", "")) / "Google/Chrome/Application/chrome.exe",
                Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application/msedge.exe"]
    browser = next((b for b in browsers if b.exists()), None)
    if not browser:
        sys.exit("no Edge or Chrome to rasterize mat.svg")
    png.unlink(missing_ok=True)
    r = subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        f"--window-size={wpx},{hpx}", f"--screenshot={png}", svg.resolve().as_uri()],
                       capture_output=True, text=True, timeout=120)
    if not png.exists():
        sys.exit(f"{browser.name} could not rasterize {svg.name}: {r.stderr.strip()[-500:]}")
    print(f"texture: {png} ({wpx} x {hpx} px)")


if __name__ == "__main__":
    main()
