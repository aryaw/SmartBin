import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.detector import detect_image, detect_video
from app.config import ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT


def main():
    parser = argparse.ArgumentParser(description="SmartBin inference helper")
    parser.add_argument("input", type=str, help="Path to image or video file")
    parser.add_argument("--model", type=str, default=None, help="Override model path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)

    ext = input_path.suffix.lower()
    if ext in ALLOWED_IMAGE_EXT:
        result = detect_image(input_path)
        print(f"Detected {result['summary']['total']} objects")
        print(f"  Organik: {result['summary']['organik']}")
        print(f"  Non-Organik: {result['summary']['non_organik']}")
        print(f"Result: {result['result_url']}")
    elif ext in ALLOWED_VIDEO_EXT:
        result = detect_video(input_path)
        print(f"Frames processed: {result['frames_processed']}")
        print(f"Detected {result['summary']['total']} objects")
        print(f"  Organik: {result['summary']['organik']}")
        print(f"  Non-Organik: {result['summary']['non_organik']}")
        print(f"Result: {result['result_url']}")
    else:
        print(f"Error: unsupported format {ext}")
        sys.exit(1)


if __name__ == "__main__":
    main()
