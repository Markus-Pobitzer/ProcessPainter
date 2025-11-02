"""Utiltiy functions for images."""

from typing import Tuple, Union

from PIL import Image, ImageOps


def pil_resize(
    image: Image.Image,
    target_size: Tuple[int, int],
    pad_input: bool = False,
    padding_color: Union[str, int, Tuple[int, ...]] = "white",
) -> Image.Image:
    """Resizing it to the target size.

    Args:
        image: Input image to be processed.
        target_size: Target size (width, height).
        pad_input: If set resizes the image while keeping the aspect ratio and pads the unfilled part.
        padding_color: The color for the padded pixels.

    Returns:
        The resized image
    """
    if pad_input:
        # Resize image, keep aspect ratio
        image = ImageOps.contain(image, size=target_size)
        # Pad while keeping image in center
        image = ImageOps.pad(image, size=target_size, color=padding_color)
    else:
        image = image.resize(target_size)
    return image


def undo_pil_resize(
    image: Image.Image,
    target_size: Tuple[int, int],
) -> Image.Image:
    """Undo the resizing and padding of the input image to the a new image with size target_size.

    Args:
        image: Input image to be processed.
        target_size: Target size (width, height).

    Returns:
        The resized image
    """
    tmp_img = Image.new(mode="RGB", size=target_size)
    # Get the resized image size
    tmp_img = ImageOps.contain(tmp_img, size=image.size)

    # Undo padding by center cropping
    width, height = image.size
    tmp_width, tmp_height = tmp_img.size

    left = int(round((width - tmp_width) / 2.0))
    top = int(round((height - tmp_height) / 2.0))
    right = left + tmp_width
    bottom = top + tmp_height
    cropped = image.crop((left, top, right, bottom))

    # Undo resizing
    ret = cropped.resize(target_size)
    return ret
