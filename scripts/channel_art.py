"""The channel's avatar and banner, in the same dark language as the video thumbnails.

    python scripts/channel_art.py --name "Channel Name" [--tagline "..."] [--backdrop FILE]

Two files into docs/channel/: avatar.png (800x800) and banner.png (2560x1440).

The look is taken from pipeline/thumbnail.py rather than invented next to it: the same Anton
face, the same white-over-amber pairing, the same raised contrast, the same vignette, the same
scrim behind the type. A channel whose banner belongs to a different design than its thumbnails
reads as two channels.

Two constraints decide the geometry, and neither is negotiable:

- A banner is cropped differently on every device. Only the middle 1546x423 of 2560x1440 is
  shown everywhere, so everything that must be read lives inside it, and the darkening that
  makes the type readable is spread across the full width - otherwise a desktop crop shows a
  bright edge the safe area never warned about.
- An avatar is shown as a circle, usually at 48 pixels. So nothing meaningful goes near the
  corners, and the monogram is set large enough to survive being shrunk to a thumbnail of
  itself - which the script measures rather than assumes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = ROOT / "assets" / "fonts" / "Anton-Regular.ttf"
OUT_DIR = ROOT / "docs" / "channel"

# Straight from pipeline/thumbnail.py, so the three pieces match.
WHITE = (255, 255, 255)
WARM = (240, 176, 84)
CONTRAST = 1.30
VIGNETTE_STRENGTH = 0.70

BANNER = (2560, 1440)
SAFE = (1546, 423)               # what every device shows, centred
BANNER_DIM = 0.55                # how far the backdrop is pushed down before the type goes on
BAND_DARKNESS = 0.78             # the band behind the type, at its strongest
BAND_FEATHER = 260               # how far it fades above and below the safe area

AVATAR = 800
AVATAR_MIN_PX = 48               # the size an avatar is actually read at
AVATAR_MIN_CAP_PX = 13           # a capital of the monogram, at that size


def vignette(size, strength=VIGNETTE_STRENGTH):
    w, h = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.clip(np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
                / np.sqrt(2), 0, 1)
    return 1 - strength * (r * r * (3 - 2 * r))


def fill_crop(img, size):
    """Cover `size` without distorting: scale to fill, then take the middle."""
    w, h = size
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(w, int(img.width * scale + 0.5)),
                      max(h, int(img.height * scale + 0.5))), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def quietest(paths):
    """The calmest picture of the set: dark, and without detail where type will sit.

    A banner carries more words than a thumbnail and they sit across the middle, so what is
    wanted here is the opposite of a thumbnail background - not the most striking picture,
    the one that argues least with type.
    """
    best, best_score = None, None
    for path in paths:
        try:
            with Image.open(path) as im:
                grey = np.asarray(im.convert("L").resize((160, 90), Image.LANCZOS),
                                  dtype=np.float64)
        except Exception:                                        # noqa: BLE001
            continue
        # The rows the type crosses, and within them the columns the safe area covers:
        # 1546 of 2560 is the middle 60%. A busy subject standing in the centre of the
        # frame is exactly what must not be chosen, however good it looked in a thumbnail.
        middle = grey[26:64]
        centre = middle[:, 32:128]
        score = (centre.mean()
                 + 5 * np.abs(np.diff(centre, axis=1)).mean()
                 + 2 * np.abs(np.diff(middle, axis=1)).mean())
        if best_score is None or score < best_score:
            best, best_score = path, score
    return best


def atmosphere(size, seed=7):
    """The thumbnails' palette with nothing in it: cold near-black, a warm light low left.

    Not a frame from a video, on purpose. Every candidate frame carried the thing the picture
    rules exist to keep out - a sign with invented lettering, a screen, a sheet of paper - and
    a banner is the one image on the channel that is never redrawn. Built from the palette
    instead, it cannot acquire a defect later, it survives any crop a device chooses, and it
    is still unmistakably the same hand as the thumbnails: the same darkness, the same cold
    blue, the same amber the type uses.
    """
    w, h = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    nx, ny = x / w, y / h

    img = np.zeros((h, w, 3), dtype=np.float64)
    img += np.array([9.0, 12.0, 17.0])                           # cold near-black

    def glow(cx, cy, radius, colour, strength):
        d2 = ((nx - cx) ** 2 + ((ny - cy) * (h / w) * 2.2) ** 2) / (radius ** 2)
        return np.array(colour, dtype=np.float64) * strength * np.exp(-d2)[:, :, None]

    img += glow(0.16, 0.86, 0.46, WARM, 0.52)                    # the warm lamp, low and left
    img += glow(0.84, 0.30, 0.52, (70, 120, 150), 0.30)          # the cold side of the frame
    img += glow(0.50, 0.55, 0.80, (24, 34, 46), 0.55)            # the room the type sits in

    # A faint horizontal lift, the way a platform or a corridor floor catches light.
    img += np.array([16.0, 18.0, 22.0]) * np.exp(
        -((ny - 0.72) ** 2) / (2 * 0.06 ** 2))[:, :, None]

    img *= vignette(size, 0.62)[:, :, None]
    img += rng.normal(0, 2.2, (h, w, 1))                         # grain, so it is not plastic
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def backdrop(path, size):
    """The picture, treated the way a thumbnail background is: contrast up, then down to dark."""
    with Image.open(path) as im:
        img = fill_crop(im.convert("RGB"), size)
    img = ImageEnhance.Contrast(img).enhance(CONTRAST)
    arr = np.asarray(img, dtype=np.float64) * (1 - BANNER_DIM)
    arr *= vignette(size)[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def band(img, top, bottom, feather=BAND_FEATHER, darkness=BAND_DARKNESS):
    """Darken a horizontal band across the whole width, fading out above and below.

    Across the whole width on purpose: the safe area is the only part shown everywhere, but
    the parts beside it ARE shown on a desktop, and a band that stopped at the safe edge would
    draw a rectangle on exactly the screens that see most of the picture.
    """
    h = img.height
    y = np.arange(h, dtype=np.float64)
    above = np.clip((y - (top - feather)) / feather, 0, 1)
    below = np.clip(((bottom + feather) - y) / feather, 0, 1)
    mask = np.minimum(above, below)
    mask = mask * mask * (3 - 2 * mask)                          # smoothstep, no banding
    arr = np.asarray(img, dtype=np.float64) * (1 - darkness * mask)[:, None, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def fitted(text, width, ceiling, font_path=FONT_PATH):
    """The largest Anton size at which `text` fits `width`."""
    size = ceiling
    while size > 10:
        font = ImageFont.truetype(str(font_path), size)
        if font.getbbox(text)[2] - font.getbbox(text)[0] <= width:
            return font, size
        size -= 2
    return ImageFont.truetype(str(font_path), 10), 10


def draw_centred(img, text, font, centre_y, colour):
    """Draw `text` centred on the canvas, its optical middle at centre_y. Returns its box."""
    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x = (img.width - w) // 2 - box[0]
    y = int(centre_y - h / 2) - box[1]
    draw.text((x, y), text, font=font, fill=colour)
    return (x + box[0], y + box[1], x + box[0] + w, y + box[1] + h)


def make_banner(name, tagline, source, out_path):
    img = backdrop(source, BANNER) if source else atmosphere(BANNER)
    safe_top = (BANNER[1] - SAFE[1]) // 2
    safe_bottom = safe_top + SAFE[1]
    img = band(img, safe_top, safe_bottom)

    name = name.upper()
    tagline = tagline.upper()
    # Inside the safe area, with room to breathe: the words may never touch its edge, because
    # a device that crops a little tighter than the documented box would cut a letter in half.
    inner = int(SAFE[0] * 0.88)
    name_font, name_size = fitted(name, inner, int(SAFE[1] * 0.46))
    tag_font, _ = fitted(tagline, int(inner * 0.80), max(30, int(name_size * 0.34)))

    gap = int(name_size * 0.40)
    name_h = name_font.getbbox(name)[3] - name_font.getbbox(name)[1]
    tag_h = tag_font.getbbox(tagline)[3] - tag_font.getbbox(tagline)[1]
    block_top = (BANNER[1] - (name_h + gap + tag_h)) / 2
    draw_centred(img, name, name_font, block_top + name_h / 2, WHITE)
    rule_y = int(block_top + name_h + gap * 0.46)
    draw_centred(img, tagline, tag_font, block_top + name_h + gap + tag_h / 2, WARM)

    # A thin amber rule between the two lines, the same accent the thumbnails use.
    draw = ImageDraw.Draw(img)
    half = int(SAFE[0] * 0.14)
    draw.rectangle([BANNER[0] // 2 - half, rule_y, BANNER[0] // 2 + half, rule_y + 4], fill=WARM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return img


def initials(name):
    """Up to three letters: what a monogram can show at 48 pixels and still be read."""
    words = [w for w in "".join(c if c.isalnum() or c.isspace() else " "
                                for c in name).split() if w]
    if len(words) == 1:
        return words[0][:2].upper()
    return "".join(w[0] for w in words[:3]).upper()


def make_avatar(name, out_path):
    size = AVATAR
    # Near-black, with the warm light the thumbnails always have somewhere low in the frame.
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    glow = np.exp(-(((x - size * 0.30) ** 2 + (y - size * 0.78) ** 2) / (2 * (size * 0.34) ** 2)))
    base = np.zeros((size, size, 3), dtype=np.float64)
    base += np.array([13, 16, 21])
    base += np.array(WARM, dtype=np.float64) * 0.30 * glow[:, :, None]
    base *= vignette((size, size), 0.55)[:, :, None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    mark = initials(name)
    # Sized to the circle, not the square: the corners are never shown.
    font, _ = fitted(mark, int(size * 0.62), int(size * 0.60))
    box = draw_centred(img, mark, font, size * 0.455, WHITE)
    draw = ImageDraw.Draw(img)
    rule_w = int((box[2] - box[0]) * 0.82)
    rule_y = int(size * 0.705)
    draw.rectangle([(size - rule_w) // 2, rule_y, (size + rule_w) // 2, rule_y + int(size * 0.035)],
                   fill=WARM)

    cap = font.getbbox("H")
    cap_px = (cap[3] - cap[1]) * AVATAR_MIN_PX / size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return img, cap_px


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="the channel name, as it is shown")
    parser.add_argument("--tagline", default="First millions and Stoic discipline")
    parser.add_argument("--backdrop", default="", help="a picture for the banner; by "
                                                       "default none, and the backdrop is "
                                                       "built from the palette")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    out = Path(args.out)
    source = Path(args.backdrop) if args.backdrop else None
    print(f"banner backdrop: {source or 'built from the palette, no photograph'}")

    banner = make_banner(args.name, args.tagline, source, out / "banner.png")
    avatar, cap_px = make_avatar(args.name, out / "avatar.png")
    print(f"banner {banner.size[0]}x{banner.size[1]}, "
          f"{(out / 'banner.png').stat().st_size / 1024:.0f} KB (YouTube allows 6 MB)")
    print(f"avatar {avatar.size[0]}x{avatar.size[1]}, "
          f"{(out / 'avatar.png').stat().st_size / 1024:.0f} KB (YouTube allows 4 MB), "
          f"monogram '{initials(args.name)}' is {cap_px:.1f}px tall at {AVATAR_MIN_PX}px "
          f"(needs {AVATAR_MIN_CAP_PX})")
    if cap_px < AVATAR_MIN_CAP_PX:
        print("WARNING: the monogram would not be readable at avatar size")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
