from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.dataset_pairs import (
    PairFeatureDataset,
    build_pair_examples,
    discover_dataset_dirs,
    load_many_dataset_records,
    split_examples,
)
from models.feature_extractor import ClipEmbeddingExtractor, build_directional_pair_features
from models.pairwise_model import PairwiseTransitionMLP, save_transition_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a directional comic page transition model P(A -> B)."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset input. Accepts a metadata dataset dir, a comic dir, "
            "datasets/index.json, or the datasets root. Repeat as needed."
        ),
    )
    parser.add_argument("--cache-dir", default=".cache/clip_embeddings")
    parser.add_argument("--output", default="outputs/learned_relation_model.pt")
    parser.add_argument("--metrics-output", default="outputs/learned_relation_metrics.json")
    parser.add_argument("--clip-backend", default="auto", choices=["auto", "clip", "openai_clip", "transformers", "transformers_clip"])
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--transformers-clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )

    dataset_dirs = discover_dataset_dirs(args.dataset)
    records = load_many_dataset_records(dataset_dirs)
    if not records:
        raise RuntimeError("No image records were loaded from the requested datasets.")

    examples = build_pair_examples(
        records,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError("No pair examples were generated. Check metadata original_order/source_index.")

    train_examples, validation_examples = split_examples(
        examples,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    clip_extractor = ClipEmbeddingExtractor(
        model_name=args.clip_model,
        transformers_model_name=args.transformers_clip_model,
        backend=args.clip_backend,
        device=str(device),
        cache_dir=args.cache_dir,
    )
    unique_paths = sorted({str(record.image_path) for record in records})
    embeddings_by_path = clip_extractor.extract_many(unique_paths)
    records_by_key = {record.page_key: record for record in records}
    embedding_dim = len(next(iter(embeddings_by_path.values())))

    train_dataset = PairFeatureDataset(train_examples, records_by_key, embeddings_by_path)
    validation_dataset = PairFeatureDataset(validation_examples, records_by_key, embeddings_by_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    model = PairwiseTransitionMLP(embedding_dim=embedding_dim, dropout=args.dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    best_metrics: dict[str, float] = {}
    best_state_dict = None
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        validation_loss, validation_accuracy = evaluate_pairs(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )
        ranking_accuracy = evaluate_ranking(
            model=model,
            examples=validation_examples,
            records_by_key=records_by_key,
            embeddings_by_path=embeddings_by_path,
            device=device,
        )
        metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "ranking_accuracy": ranking_accuracy,
        }
        history.append(metrics)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f} "
            f"rank_acc={ranking_accuracy:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_metrics = metrics
            best_state_dict = deepcopy(model.state_dict())
            save_transition_checkpoint(
                path=args.output,
                model=model,
                optimizer=optimizer,
                embedding_dim=embedding_dim,
                metrics=best_metrics,
                dropout=args.dropout,
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    scenario_metrics = evaluate_by_scenario(
        model=model,
        examples=validation_examples,
        records_by_key=records_by_key,
        embeddings_by_path=embeddings_by_path,
        device=device,
        batch_size=args.batch_size,
    )

    report = {
        "inputs": [str(Path(dataset).resolve()) for dataset in args.dataset],
        "datasets": [str(dataset_dir) for dataset_dir in dataset_dirs],
        "records": len(records),
        "pairs": len(examples),
        "train_pairs": len(train_examples),
        "validation_pairs": len(validation_examples),
        "embedding_dim": embedding_dim,
        "pair_feature_dim": embedding_dim * 4,
        "clip_backend": clip_extractor.loaded_backend,
        "best_metrics": best_metrics,
        "scenario_metrics": scenario_metrics,
        "history": history,
    }
    metrics_path = Path(args.metrics_output).expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint: {Path(args.output).resolve()}")
    print(f"Saved metrics: {metrics_path}")


def train_one_epoch(
    model: PairwiseTransitionMLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        correct += int(((predictions >= 0.5) == (labels >= 0.5)).sum().item())
        total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def evaluate_pairs(
    model: PairwiseTransitionMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            predictions = model(features)
            loss = criterion(predictions, labels)
            total_loss += float(loss.item()) * labels.size(0)
            correct += int(((predictions >= 0.5) == (labels >= 0.5)).sum().item())
            total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def evaluate_ranking(
    model: PairwiseTransitionMLP,
    examples,
    records_by_key,
    embeddings_by_path,
    device: torch.device,
) -> float:
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    negative_by_source = {}
    for negative in negatives:
        negative_by_source.setdefault(negative.source_key, []).append(negative)
    if not positives:
        return 0.0

    wins = 0
    comparisons = 0
    model.eval()
    with torch.no_grad():
        for positive in positives:
            source_negatives = negative_by_source.get(positive.source_key) or negatives
            if not source_negatives:
                continue
            negative = source_negatives[0]
            positive_score = _score_example(
                model, positive, records_by_key, embeddings_by_path, device
            )
            negative_score = _score_example(
                model, negative, records_by_key, embeddings_by_path, device
            )
            wins += int(positive_score > negative_score)
            comparisons += 1
    return wins / max(1, comparisons)


def evaluate_by_scenario(
    model: PairwiseTransitionMLP,
    examples,
    records_by_key,
    embeddings_by_path,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    grouped = {}
    for example in examples:
        source = records_by_key[example.source_key]
        grouped.setdefault(source.dataset, []).append(example)

    metrics = {}
    criterion = nn.BCELoss()
    for dataset_name, dataset_examples in sorted(grouped.items()):
        dataset = PairFeatureDataset(dataset_examples, records_by_key, embeddings_by_path)
        loader = DataLoader(dataset, batch_size=batch_size)
        loss, accuracy = evaluate_pairs(model, loader, criterion, device)
        ranking_accuracy = evaluate_ranking(
            model=model,
            examples=dataset_examples,
            records_by_key=records_by_key,
            embeddings_by_path=embeddings_by_path,
            device=device,
        )
        positives = sum(1 for example in dataset_examples if example.label == 1)
        negatives = len(dataset_examples) - positives
        metrics[dataset_name] = {
            "examples": float(len(dataset_examples)),
            "positives": float(positives),
            "negatives": float(negatives),
            "loss": loss,
            "accuracy": accuracy,
            "ranking_accuracy": ranking_accuracy,
        }
    return metrics


def _score_example(
    model: PairwiseTransitionMLP,
    example,
    records_by_key,
    embeddings_by_path,
    device: torch.device,
) -> float:
    source = records_by_key[example.source_key]
    target = records_by_key[example.target_key]
    features = build_directional_pair_features(
        embeddings_by_path[str(source.image_path)],
        embeddings_by_path[str(target.image_path)],
    )
    prediction = model(features.unsqueeze(0).to(device))
    return float(prediction.squeeze().detach().cpu().item())


if __name__ == "__main__":
    main()
