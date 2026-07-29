"""Create a paper-ready horizontal strip from the three 2x3 WSI panels."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
SOURCES = [
    ROOT / "WSI_choosed.png",
    ROOT / "WSI_choosed2.png",
    ROOT / "WSI_choosed3.png",
]
OUT = ROOT / "WSI_choosed_horizontal_strip_with_labels.png"

# These dimensions make three 3-column groups form one compact, wide strip.
CELL_W, CELL_H = 300, 225
GROUP_GAP = 16
LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)


def fit_on_white(panel: Image.Image) -> Image.Image:
    """Resize without distortion and centre the panel in a uniform cell."""
    panel = panel.convert("RGB")
    scale = min(CELL_W / panel.width, CELL_H / panel.height)
    size = (round(panel.width * scale), round(panel.height * scale))
    panel = panel.resize(size, Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (CELL_W, CELL_H), "white")
    cell.paste(panel, ((CELL_W - size[0]) // 2, (CELL_H - size[1]) // 2))
    return cell


def make_group(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    group = Image.new("RGB", (CELL_W * 3, CELL_H * 2), "white")
    for row in range(2):
        y0, y1 = round(row * image.height / 2), round((row + 1) * image.height / 2)
        for col in range(3):
            x0, x1 = round(col * image.width / 3), round((col + 1) * image.width / 3)
            panel = fit_on_white(image.crop((x0, y0, x1, y1)))
            group.paste(panel, (col * CELL_W, row * CELL_H))
    draw = ImageDraw.Draw(group)
    for index, label in enumerate(LABELS):
        row, col = divmod(index, 3)
        x, y = col * CELL_W + 7, (row + 1) * CELL_H - 24
        # A small white backing keeps the label legible over dense tissue.
        bbox = draw.textbbox((x, y), label, font=FONT)
        draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill="white")
        draw.text((x, y), label, font=FONT, fill="black")
    return group


groups = [make_group(source) for source in SOURCES]
canvas = Image.new(
    "RGB", (len(groups) * CELL_W * 3 + (len(groups) - 1) * GROUP_GAP, CELL_H * 2), "white"
)
for index, group in enumerate(groups):
    canvas.paste(group, (index * (CELL_W * 3 + GROUP_GAP), 0))

canvas.save(OUT, optimize=True)
print(OUT)
