"""Create a short demo GIF from the existing screenshots."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "screenshots"
OUTPUT = SCREENSHOTS / "demo.gif"

FRAMES = [
    ("home.png", "1. Upload two historical texts"),
    ("results.png", "2. Compare similarity, terms and themes"),
    ("charts.png", "3. Review charts"),
    ("propaganda.png", "4. Compare rule-based and ML propaganda signals"),
    ("highlight.png", "5. Inspect highlighted evidence"),
]


def main() -> None:
    frames: list[Image.Image] = []
    for filename, caption in FRAMES:
        image_path = SCREENSHOTS / filename
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        image.thumbnail((1280, 760))
        canvas = Image.new("RGB", (1280, 820), "#f3eddc")
        x = (canvas.width - image.width) // 2
        canvas.paste(image, (x, 20))

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=26)
        draw.rounded_rectangle((28, 766, 1252, 806), radius=8, fill="#2f2a24")
        draw.text((48, 776), caption, fill="#fff8f1", font=font)
        frames.append(canvas)

    if not frames:
        raise RuntimeError("No screenshots found for GIF generation.")

    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=1300,
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    main()
