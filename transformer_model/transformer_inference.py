from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from utils import tokenize_with_offsets


@dataclass
class NERPrediction:
    entities: List[Dict[str, str]]
    tags: List[str]
    tokens: List[str]


DEFAULT_NER_TAGS = [
    "<PAD>",
    "O",
    "B-TEAM", "I-TEAM",
    "B-STADIUM", "I-STADIUM",
    "B-PLAYER", "I-PLAYER",
    "B-COACH", "I-COACH",
]


def load_enhanced_metadata(checkpoint_path: str) -> Dict[str, object]:
    metadata_path = Path(checkpoint_path) / "enhanced_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing enhanced metadata file: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def decode_bio_predictions(
    text: str,
    token_spans: Sequence[Tuple[str, int, int]],
    tags: Sequence[str],
) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    seen = set()

    current_label: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    def flush_current_entity() -> None:
        nonlocal current_label, current_start, current_end
        if current_label is None or current_start is None or current_end is None:
            current_label = None
            current_start = None
            current_end = None
            return

        start_char = token_spans[current_start][1]
        end_char = token_spans[current_end][2]
        entity_text = text[start_char:end_char].strip()
        if entity_text:
            key = (entity_text, current_label)
            if key not in seen:
                entities.append({"text": entity_text, "label": current_label})
                seen.add(key)

        current_label = None
        current_start = None
        current_end = None

    for idx, tag in enumerate(tags):
        if tag in {"O", "<PAD>"}:
            flush_current_entity()
            continue

        if "-" not in tag:
            flush_current_entity()
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "B":
            flush_current_entity()
            current_label = label
            current_start = idx
            current_end = idx
            continue

        if prefix == "I" and current_label == label and current_end is not None:
            current_end = idx
            continue

        flush_current_entity()
        if prefix == "I":
            current_label = label
            current_start = idx
            current_end = idx

    flush_current_entity()
    return entities


class TransformerNERInferencePipeline:
    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.metadata = load_enhanced_metadata(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint_path).to(device)
        self.model.eval()
        self.device = device
        self.id2tag = {
            int(key): str(value)
            for key, value in self.metadata.get("id2tag", {}).items()
        }
        if not self.id2tag:
            self.id2tag = {idx: tag for idx, tag in enumerate(DEFAULT_NER_TAGS)}
        self.max_length = int(self.metadata.get("max_length", 256))

    def predict(self, text: str) -> NERPrediction:
        token_spans = tokenize_with_offsets(text)
        if not token_spans:
            return NERPrediction(entities=[], tags=[], tokens=[])

        tokens = [token for token, _, _ in token_spans]
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits[0]

        pred_ids = logits.argmax(dim=-1).tolist()
        tokenizer_output = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_id_sequence = tokenizer_output.word_ids()

        word_predictions: Dict[int, int] = {}
        for token_idx, word_id in enumerate(word_id_sequence):
            if word_id is None:
                continue
            if word_id not in word_predictions:
                word_predictions[word_id] = pred_ids[token_idx]

        tags = []
        for word_index in range(len(tokens)):
            pred_id = word_predictions.get(word_index, 1)
            tags.append(self.id2tag.get(int(pred_id), "O"))

        entities = decode_bio_predictions(text=text, token_spans=token_spans, tags=tags)
        return NERPrediction(entities=entities, tags=tags, tokens=tokens)