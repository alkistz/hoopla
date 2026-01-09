#!/usr/bin/env python3
"""CLI for multimodal search operations."""

import sys
from pathlib import Path

# Add parent directory to path to import from lib
sys.path.insert(0, str(Path(__file__).parent))

from lib.multimodal_search import image_search_command, verify_image_embedding


def main():
    if len(sys.argv) < 2:
        print("Usage: python multimodal_search_cli.py <command> [args...]")
        print("Available commands: verify_image_embedding, image_search")
        sys.exit(1)

    command = sys.argv[1]

    if command == "verify_image_embedding":
        if len(sys.argv) != 3:
            print(
                "Usage: python multimodal_search_cli.py verify_image_embedding <image_path>"
            )
            sys.exit(1)
        image_path = sys.argv[2]
        verify_image_embedding(image_path)
    elif command == "image_search":
        if len(sys.argv) != 3:
            print("Usage: python multimodal_search_cli.py image_search <image_path>")
            sys.exit(1)
        image_path = sys.argv[2]
        results = image_search_command(image_path)

        # Print results in the specified format
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
            print(f"   {result['description'][:100]}...")
            if i < len(results):
                print()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: verify_image_embedding, image_search")
        sys.exit(1)


if __name__ == "__main__":
    main()
