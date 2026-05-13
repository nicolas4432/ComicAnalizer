from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.dataset_pairs import load_many_dataset_records
from models.feature_extractor import ClipEmbeddingExtractor, build_directional_pair_features
from models.pairwise_model import LearnedRelationModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a learned transition model with full-candidate ranking."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--clip-backend", default="clip")
    parser.add_argument("--cache-dir", default=".cache/clip_embeddings")
    parser.add_argument("--output", default="outputs/learned_relation_ranking_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_many_dataset_records(args.dataset)

    clip_extractor = ClipEmbeddingExtractor(
        backend=args.clip_backend,
        cache_dir=args.cache_dir,
    )
    learned_model = LearnedRelationModel(
        checkpoint_path=args.checkpoint,
        clip_extractor=clip_extractor,
    )
    unique_paths = sorted({str(record.image_path) for record in records})
    embeddings_by_path = clip_extractor.extract_many(unique_paths)

    by_dataset = {}
    for source in records:
        if source.page_type != "comic_page" or source.source_index is None:
            continue
        candidates = [
            record
            for record in records
            if record.dataset == source.dataset and record.page_key != source.page_key
        ]
        if not candidates:
            continue
        correct_keys = {
            candidate.page_key
            for candidate in candidates
            if candidate.page_type == "comic_page"
            and candidate.comic_id == source.comic_id
            and candidate.source_index == source.source_index + 1
        }
        if not correct_keys:
            continue
        scored = []
        for candidate in candidates:
            features = build_directional_pair_features(
                embeddings_by_path[str(source.image_path)],
                embeddings_by_path[str(candidate.image_path)],
            )
            with torch.no_grad():
                score = learned_model.model(
                    features.unsqueeze(0).to(learned_model.device)
                )
            scored.append((candidate.page_key, float(score.squeeze().cpu().item())))
        scored.sort(key=lambda item: item[1], reverse=True)
        rank = 1 + next(
            index for index, (page_key, _) in enumerate(scored) if page_key in correct_keys
        )
        bucket = by_dataset.setdefault(
            source.dataset,
            {"ranks": [], "top1": 0, "top3": 0, "top5": 0, "total": 0},
        )
        bucket["ranks"].append(rank)
        bucket["top1"] += int(rank == 1)
        bucket["top3"] += int(rank <= 3)
        bucket["top5"] += int(rank <= 5)
        bucket["total"] += 1

    metrics = {}
    total_top1 = total_top3 = total_top5 = total = 0
    for dataset_name, bucket in sorted(by_dataset.items()):
        count = max(1, bucket["total"])
        metrics[dataset_name] = {
            "transitions": bucket["total"],
            "top1_accuracy": bucket["top1"] / count,
            "top3_accuracy": bucket["top3"] / count,
            "top5_accuracy": bucket["top5"] / count,
            "mean_rank": sum(bucket["ranks"]) / count,
            "median_rank": sorted(bucket["ranks"])[len(bucket["ranks"]) // 2],
        }
        total_top1 += bucket["top1"]
        total_top3 += bucket["top3"]
        total_top5 += bucket["top5"]
        total += bucket["total"]

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "datasets": [str(Path(dataset).resolve()) for dataset in args.dataset],
        "overall": {
            "transitions": total,
            "top1_accuracy": total_top1 / max(1, total),
            "top3_accuracy": total_top3 / max(1, total),
            "top5_accuracy": total_top5 / max(1, total),
        },
        "by_dataset": metrics,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
