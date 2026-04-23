from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _bio_to_entities(tags: Sequence[str]) -> Set[Tuple[int, int, str]]:
    entities: Set[Tuple[int, int, str]] = set()
    current_label: str | None = None
    start_index: int | None = None

    def close_entity(end_index: int) -> None:
        nonlocal current_label, start_index
        if current_label is not None and start_index is not None:
            entities.add((start_index, end_index, current_label))
        current_label = None
        start_index = None

    for index, tag in enumerate(tags):
        if not tag or tag == "<PAD>" or tag == "O":
            close_entity(index - 1)
            continue

        if "-" not in tag:
            close_entity(index - 1)
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "B":
            close_entity(index - 1)
            current_label = label
            start_index = index
        elif prefix == "I":
            if current_label != label or start_index is None:
                close_entity(index - 1)
                current_label = label
                start_index = index
        else:
            close_entity(index - 1)

    close_entity(len(tags) - 1)
    return entities


def compute_ner_metrics(
    true_tag_sequences: Sequence[Sequence[str]],
    pred_tag_sequences: Sequence[Sequence[str]],
) -> Dict[str, float]:
    if len(true_tag_sequences) != len(pred_tag_sequences):
        raise ValueError("true_tag_sequences and pred_tag_sequences must have the same length.")

    correct_tokens = 0
    total_tokens = 0
    tp = 0
    fp = 0
    fn = 0

    for true_tags, pred_tags in zip(true_tag_sequences, pred_tag_sequences):
        seq_len = min(len(true_tags), len(pred_tags))

        for true_tag, pred_tag in zip(true_tags[:seq_len], pred_tags[:seq_len]):
            correct_tokens += int(true_tag == pred_tag)
            total_tokens += 1

        true_entities = _bio_to_entities(true_tags[:seq_len])
        pred_entities = _bio_to_entities(pred_tags[:seq_len])

        tp += len(true_entities & pred_entities)
        fp += len(pred_entities - true_entities)
        fn += len(true_entities - pred_entities)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "token_accuracy": _safe_div(correct_tokens, total_tokens),
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
    }
