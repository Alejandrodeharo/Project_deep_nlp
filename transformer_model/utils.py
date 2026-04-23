from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


NER_TAGS = [
    "<PAD>",
    "O",
    "B-TEAM", "I-TEAM",
    "B-STADIUM", "I-STADIUM",
    "B-PLAYER", "I-PLAYER",
    "B-COACH", "I-COACH",
]

_ALLOWED_NER_LABELS = {"TEAM", "STADIUM", "PLAYER", "COACH"}
_TOKEN_PATTERN = r"\w+(?:[-'’]\w+)*|[^\w\s]"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_json(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, but got {type(data).__name__}")
    return data


def normalize_sentiment_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for example in examples:
        item = dict(example)

        if "text" not in item:
            raise KeyError("Each example must contain 'text'.")

        if "label" not in item:
            if "sentiment" in item:
                item["label"] = item["sentiment"]
            else:
                raise KeyError("Each sentiment example must contain either 'label' or 'sentiment'.")

        normalized.append(item)

    return normalized


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in re.finditer(_TOKEN_PATTERN, text)]


def find_all_occurrences(text: str, substring: str) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(substring)))
        start = idx + 1
    return matches


def build_bio_tags(example: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    text = example["text"]
    entities = example.get("entities", [])

    token_spans = tokenize_with_offsets(text)
    tokens = [token for token, _, _ in token_spans]
    tags = ["O"] * len(tokens)

    entities_sorted = sorted(entities, key=lambda entity: len(str(entity.get("text", ""))), reverse=True)

    for entity in entities_sorted:
        entity_text = str(entity.get("text", "")).strip()
        entity_label = str(entity.get("label", "")).strip().upper()

        if not entity_text or entity_label not in _ALLOWED_NER_LABELS:
            continue

        occurrences = find_all_occurrences(text, entity_text)

        for entity_start, entity_end in occurrences:
            overlapping_token_ids: List[int] = []

            for index, (_, token_start, token_end) in enumerate(token_spans):
                if token_start < entity_end and token_end > entity_start:
                    overlapping_token_ids.append(index)

            if not overlapping_token_ids:
                continue

            if any(tags[index] != "O" for index in overlapping_token_ids):
                continue

            tags[overlapping_token_ids[0]] = f"B-{entity_label}"
            for index in overlapping_token_ids[1:]:
                tags[index] = f"I-{entity_label}"

    return tokens, tags
