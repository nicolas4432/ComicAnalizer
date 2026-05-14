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
        description="Create a Magi ZIP from clean by_comic datasets for cloud tests."
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
        default="outputs/packages/magi_cloud_sample",
        help="Folder where the cloud dataset will be assembled.",
    )
    parser.add_argument(
        "--zip-path",
        default="outputs/packages/magi_cloud_sample.zip",
        help="ZIP path to create for Colab upload.",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Copy every page from each selected clean dataset instead of one sampled page.",
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
        help="Maximum comics to include. 0 means all comics.",
    )
    parser.add_argument(
        "--comic-id",
        action="append",
        default=[],
        help="Only include this comic id. Can be repeated.",
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

    manifest: dict[str, object] = {
        "source_root": str(root),
        "dataset_name": args.dataset_name,
        "all_pages": args.all_pages,
        "selection": args.selection,
        "max_comics": args.max_comics,
        "comic_ids": args.comic_id,
        "pages": [],
    }

    copied_comics = 0
    comic_ids = set(args.comic_id or [])
    for comic_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if comic_ids and comic_dir.name not in comic_ids:
            continue
        if args.max_comics > 0 and copied_comics >= args.max_comics:
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

        selected_images = images if args.all_pages else [
            select_image(images, args.selection, args.seed, comic_dir.name)
        ]
        for selected in selected_images:
            copy_image(
                source=selected,
                output_dir=output_dir,
                comic_id=comic_dir.name,
                dataset_name=args.dataset_name,
                manifest=manifest,
                selection="all" if args.all_pages else args.selection,
            )
        copied_comics += 1

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
                "comic_count": copied_comics,
                "all_pages": args.all_pages,
                "pages": pages,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def copy_image(
    source: Path,
    output_dir: Path,
    comic_id: str,
    dataset_name: str,
    manifest: dict[str, object],
    selection: str,
) -> None:
    relative_target = Path("by_comic") / comic_id / dataset_name / source.name
    target = output_dir / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    pages = manifest["pages"]
    if isinstance(pages, list):
        pages.append(
            {
                "comic_id": comic_id,
                "dataset": dataset_name,
                "selection": selection,
                "source_path": str(source),
                "relative_path": relative_target.as_posix(),
            }
        )


if __name__ == "__main__":
    main()
