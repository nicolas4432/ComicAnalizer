from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from models.feature_extractor import build_directional_pair_features


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ComicPageRecord:
    dataset: str
    page_key: str
    image_path: Path
    page_id: str
    source_index: int | None
    comic_id: str
    page_type: str = "comic_page"


@dataclass(frozen=True)
class PairExample:
    source_key: str
    target_key: str
    label: int
    negative_type: str | None = None


def load_dataset_records(dataset_dir: str | Path) -> list[ComicPageRecord]:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_name = str(metadata.get("dataset", dataset_path.name))
    raw_records = _select_records(metadata)

    records: list[ComicPageRecord] = []
    for raw in raw_records:
        output_file = raw.get("output_file") or raw.get("clean_file")
        if not output_file:
            continue
        image_path = (dataset_path / output_file).resolve()
        if not image_path.exists() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        page_type = str(raw.get("type", "comic_page"))
        page_id = str(raw.get("page_id") or raw.get("noise_id") or output_file)
        comic_id = str(raw.get("comic_id", "comic_a"))
        source_index = raw.get("source_index")
        source_index = int(source_index) if source_index is not None else None
        page_key = f"{dataset_name}:{comic_id}:{page_id}:{output_file}"
        records.append(
            ComicPageRecord(
                dataset=dataset_name,
                page_key=page_key,
                image_path=image_path,
                page_id=page_id,
                source_index=source_index,
                comic_id=comic_id,
                page_type=page_type,
            )
        )
    return records


def load_many_dataset_records(dataset_dirs: Iterable[str | Path]) -> list[ComicPageRecord]:
    records: list[ComicPageRecord] = []
    for dataset_dir in dataset_dirs:
        records.extend(load_dataset_records(dataset_dir))
    return records


def build_pair_examples(
    records: list[ComicPageRecord],
    negatives_per_positive: int = 3,
    seed: int = 17,
) -> list[PairExample]:
    rng = random.Random(seed)
    comic_pages = [record for record in records if record.page_type == "comic_page"]
    positives = _build_positive_pairs(comic_pages)
    positive_keys = {(pair.source_key, pair.target_key) for pair in positives}
    all_by_key = {record.page_key: record for record in records}
    comic_by_key = {record.page_key: record for record in comic_pages}

    negatives: list[PairExample] = []
    for positive in positives:
        source = comic_by_key[positive.source_key]
        target = comic_by_key[positive.target_key]

        reverse = PairExample(
            source_key=target.page_key,
            target_key=source.page_key,
            label=0,
            negative_type="reverse",
        )
        if (reverse.source_key, reverse.target_key) not in positive_keys:
            negatives.append(reverse)

        negatives.extend(
            _sample_random_negatives(
                source=source,
                records=list(all_by_key.values()),
                positive_keys=positive_keys,
                rng=rng,
                count=max(0, negatives_per_positive - 1),
            )
        )

    target_negative_count = len(positives) * negatives_per_positive
    if len(negatives) > target_negative_count:
        negatives = rng.sample(negatives, target_negative_count)

    examples = positives + negatives
    rng.shuffle(examples)
    return examples


class PairFeatureDataset(Dataset):
    def __init__(
        self,
        examples: list[PairExample],
        records_by_key: dict[str, ComicPageRecord],
        embeddings_by_path: dict[str, torch.Tensor],
    ) -> None:
        self.examples = examples
        self.records_by_key = records_by_key
        self.embeddings_by_path = embeddings_by_path

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        source = self.records_by_key[example.source_key]
        target = self.records_by_key[example.target_key]
        embedding_a = self.embeddings_by_path[str(source.image_path)]
        embedding_b = self.embeddings_by_path[str(target.image_path)]
        features = build_directional_pair_features(embedding_a, embedding_b)
        label = torch.tensor([float(example.label)], dtype=torch.float32)
        return features, label


def split_examples(
    examples: list[PairExample],
    validation_ratio: float = 0.2,
    seed: int = 17,
) -> tuple[list[PairExample], list[PairExample]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio)) if len(shuffled) > 1 else 0
    return shuffled[validation_size:], shuffled[:validation_size]


def _select_records(metadata: dict) -> list[dict]:
    if metadata.get("shuffled_order"):
        return list(metadata["shuffled_order"])
    if metadata.get("kept_order"):
        return list(metadata["kept_order"])
    if metadata.get("variations"):
        source_by_page_id = {
            item["page_id"]: item for item in metadata.get("original_order", [])
        }
        variation_records: list[dict] = []
        for page_id, variants in metadata["variations"].items():
            source = source_by_page_id.get(page_id, {})
            for variant in variants:
                record = dict(variant)
                record["page_id"] = page_id
                record["source_index"] = source.get("source_index")
                record["comic_id"] = source.get("comic_id", "comic_a")
                record["type"] = "comic_page"
                variation_records.append(record)
        if variation_records:
            return variation_records
    return list(metadata.get("original_order", []))


def _build_positive_pairs(records: list[ComicPageRecord]) -> list[PairExample]:
    groups: dict[tuple[str, str], list[ComicPageRecord]] = {}
    for record in records:
        if record.source_index is None:
            continue
        groups.setdefault((record.dataset, record.comic_id), []).append(record)

    positives: list[PairExample] = []
    for grouped_records in groups.values():
        by_index: dict[int, list[ComicPageRecord]] = {}
        for record in grouped_records:
            if record.source_index is not None:
                by_index.setdefault(record.source_index, []).append(record)

        for source_index in sorted(by_index):
            next_index = source_index + 1
            if next_index not in by_index:
                continue
            for current in by_index[source_index]:
                for following in by_index[next_index]:
                    positives.append(
                        PairExample(
                            source_key=current.page_key,
                            target_key=following.page_key,
                            label=1,
                        )
                    )
    return positives


def _sample_random_negatives(
    source: ComicPageRecord,
    records: list[ComicPageRecord],
    positive_keys: set[tuple[str, str]],
    rng: random.Random,
    count: int,
) -> list[PairExample]:
    candidates = [
        candidate
        for candidate in records
        if candidate.page_key != source.page_key
        and (source.page_key, candidate.page_key) not in positive_keys
    ]
    if not candidates:
        return []

    sampled: list[PairExample] = []
    attempts = 0
    max_attempts = max(20, count * 20)
    while len(sampled) < count and attempts < max_attempts:
        attempts += 1
        candidate = rng.choice(candidates)
        negative_type = _classify_negative(source, candidate)
        example = PairExample(
            source_key=source.page_key,
            target_key=candidate.page_key,
            label=0,
            negative_type=negative_type,
        )
        if example not in sampled:
            sampled.append(example)
    return sampled


def _classify_negative(source: ComicPageRecord, target: ComicPageRecord) -> str:
    if target.page_type != "comic_page":
        return "noise"
    if source.dataset != target.dataset or source.comic_id != target.comic_id:
        return "cross_cluster"
    return "random_same_comic"
