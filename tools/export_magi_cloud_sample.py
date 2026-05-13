from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from tools.inspect_magi_dataset import select_image
from utils.images import is_supported_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small Magi sample ZIP from clean by_comic datasets."
    )
    parser.add_argument(
        "--by-comic-root",
        default=r"C:\Users\nico4\Downloads\ComicPruebas\datasets\by_comic",
        help="Root folder with datasets/by_comic/<comic>/<dataset>.",
    )
    parser.add_argument(
        "--dataset-name",
        default="test_1_clean",
        help="Dataset folder to sample from each comic.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/magi_cloud_sample",
        help="Folder where the sample dataset will be assembled.",
    )
    parser.add_argument(
        "--zip-path",
        default="outputs/magi_cloud_sample.zip",
        help="ZIP path to create for Colab upload.",
    )
    parser.add_argument(
        "--selection",
        default="middle",
        choices=["first", "middle", "last", "random"],
        help="Which page to copy from each clean comic.",
    )
    parser.add_argument(
        "--max-comics",
        type=int,
        default=8,
        help="Maximum comics to include. Use a small value for quick cloud tests.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random selection seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.by_comic_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    zip_path = Path(args.zip_path).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"by_comic root does not exist: {root}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_root = output_dir / "by_comic"
    manifest: dict[str, object] = {
        "source_root": str(root),
        "dataset_name": args.dataset_name,
        "selection": args.selection,
        "max_comics": args.max_comics,
        "pages": [],
    }

    for comic_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pages = manifest["pages"]
        if isinstance(pages, list) and args.max_comics > 0 and len(pages) >= args.max_comics:
            break

        dataset_dir = comic_dir / args.dataset_name
        if not dataset_dir.exists():
            continue

        images = sorted(
            (path for path in dataset_dir.iterdir() if is_supported_image(path)),
            key=lambda path: path.name.lower(),
        )
        if not images:
            continue

        selected = select_image(images, args.selection, args.seed, comic_dir.name)
        relative_target = Path("by_comic") / comic_dir.name / args.dataset_name / selected.name
        target = output_dir / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected, target)

        if isinstance(pages, list):
            pages.append(
                {
                    "comic_id": comic_dir.name,
                    "dataset": args.dataset_name,
                    "selection": args.selection,
                    "source_path": str(selected),
                    "relative_path": relative_target.as_posix(),
                }
            )

    pages = manifest["pages"]
    if not isinstance(pages, list) or not pages:
        raise RuntimeError(f"No images were copied from {root}")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent))

    print(
        json.dumps(
            {
                "sample_dir": str(output_dir),
                "zip_path": str(zip_path),
                "page_count": len(pages),
                "pages": pages,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
