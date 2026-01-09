#!/usr/bin/env python3
"""
Multimodal query rewriting using Gemini to convert image + text into searchable queries.
"""

import argparse
import mimetypes
from pathlib import Path

from lib.llm_setup import rewrite_query_with_image


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite a text query based on an image using Gemini"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to an image file"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query to rewrite based on the image",
    )
    args = parser.parse_args()

    # Determine MIME type of the image
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    # Read the image file
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {args.image}")
        return

    with open(image_path, "rb") as f:
        img = f.read()

    # Rewrite query using image and text
    response = rewrite_query_with_image(img, mime, args.query)

    # Print results
    rewritten_text = response.text.strip() if response.text else ""
    print(f"Rewritten query: {rewritten_text}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
