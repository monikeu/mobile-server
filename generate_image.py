import random
import uuid

from PIL import Image, ImageDraw


def generate_random_image(width=128, height=128, where='./'):
    if where[-1] != '/':
        where = where + '/'
    rand_pixels = [random.randint(0, 255) for _ in range(width * height * 3)]
    rand_pixels_as_bytes = bytes(rand_pixels)
    text_and_filename = str(uuid.uuid4())

    random_image = Image.frombytes('RGB', (width, height), rand_pixels_as_bytes)

    draw_image = ImageDraw.Draw(random_image)
    draw_image.text(xy=(0, 0), text=text_and_filename, fill=(255, 255, 255))
    random_image.save(f"{where}{text_and_filename}.png")


# Generate 42 random images:
import sys
from random import randint

from pathlib import Path

if __name__ == '__main__':
    """
    call: python -m generate_image.py images_target_directory min_size max_size samples
    """
    args = sys.argv
    call_path: str = args[0]
    where: str = args[1]
    size_min: int = int(args[2])
    size_max: int = int(args[3])
    samples: int = int(args[4])

    Path(where).mkdir(parents=True, exist_ok=True)

    for _ in range(samples):
        size = randint(size_min, size_max)
        generate_random_image(width=size, height=size, where=where)
