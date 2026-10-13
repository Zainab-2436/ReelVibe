
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

output = Path(__file__).parent / "posters" / "default.jpg"
output.parent.mkdir(exist_ok=True)

width, height = 600, 900
img = Image.new("RGB", (width, height), (12, 12, 22))
draw = ImageDraw.Draw(img)

# Cinematic gradient background
for y in range(height):
    ratio = y / height
    color = (
        int(32 - 18 * ratio),
        int(15 - 7 * ratio),
        int(40 - 15 * ratio),
    )
    draw.line((0, y, width, y), fill=color)

# Decorative cinematic lights
random.seed(12)
for _ in range(35):
    x = random.randint(0, width)
    y = random.randint(0, height)
    r = random.randint(2, 7)
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(105, 35, 53))

# Poster artwork
draw.ellipse((125, 180, 475, 530), outline=(230, 42, 65), width=8)
draw.ellipse((165, 220, 435, 490), outline=(110, 35, 57), width=3)
draw.polygon(
    [(260, 285), (260, 425), (375, 355)],
    fill=(255, 50, 75)
)

try:
    title_font = ImageFont.truetype("arial.ttf", 48)
    small_font = ImageFont.truetype("arial.ttf", 20)
except OSError:
    title_font = ImageFont.load_default()
    small_font = ImageFont.load_default()

draw.text((40, 55), "REELVIBE ORIGINAL", font=small_font, fill=(255, 80, 100))
draw.text((40, 630), "YOUR NEXT", font=title_font, fill="white")
draw.text((40, 690), "FAVORITE", font=title_font, fill="white")
draw.text((40, 750), "MOVIE", font=title_font, fill=(255, 50, 75))
draw.text((40, 835), "DISCOVER SOMETHING NEW", font=small_font, fill=(200, 195, 210))

img.save(output, quality=92)
print(f"Poster created: {output}")
