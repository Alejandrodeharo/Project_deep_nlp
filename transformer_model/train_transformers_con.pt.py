from __future__ import annotations

import argparse
import inspect
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from metrics_extended import compute_ner_metrics, compute_sentiment_metrics
from ocr_utils import append_ocr_hint_to_text, get_ocr_signal_for_record
from utils import NER_TAGS, build_bio_tags, load_json, normalize_sentiment_examples, set_seed


RESULT_LABEL_NAMES = {
    -1: "AWAY_WIN",
    0: "DRAW",
    1: "HOME_WIN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer models for football sentiment or NER.")
    parser.add_argument("--task", choices=["sentiment", "ner"], required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--use-ocr-hint", action="store_true")
    parser.add_argument("--image-dir", type=str, default="data")
    parser.add_argument("--ocr-cache-path", type=str, default="ocr_cache.json")
    return parser.parse_args()


class SentimentTransformerDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer,
        label2id: Dict[int, int],
        max_length: int,
        use_ocr_hint: bool = False,
        image_dir: str = "data",
        ocr_cache_path: str | None = None,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.use_ocr_hint = use_ocr_hint
        self.image_dir = image_dir
        self.ocr_cache_path = ocr_cache_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = dict(self.records[index])
        text = record["text"]
        if self.use_ocr_hint:
            signal = get_ocr_signal_for_record(
                record=record,
                image_dir=self.image_dir,
                cache_path=self.ocr_cache_path,
                verbose=False,
            )
            text = append_ocr_hint_to_text(text=text, record=record, signal=signal)

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = self.label2id[int(record["label"])]
        return encoded


class NERTransformerDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer,
        tag2id: Dict[str, int],
        max_length: int,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required for token classification.")

        self.items: List[Dict[str, Any]] = []
        self.word_level_tags: List[List[str]] = []

        for record in records:
            tokens, tags = build_bio_tags(record)
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
            )
            word_ids = encoded.word_ids()

            labels: List[int] = []
            visible_tags: List[str] = []
            previous_word_id = None
            seen_word_ids = set()

            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                    continue
                if word_id != previous_word_id:
                    tag = tags[word_id]
                    labels.append(tag2id[tag])
                    previous_word_id = word_id
                    if word_id not in seen_word_ids:
                        visible_tags.append(tag)
                        seen_word_ids.add(word_id)
                else:
                    labels.append(-100)

            encoded["labels"] = labels
            self.items.append(dict(encoded))
            self.word_level_tags.append(visible_tags)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


def choose_default_model(task: str) -> str:
    if task == "sentiment":
        return "distilroberta-base"
    return "bert-base-cased"


def stratified_split(
    records: Sequence[Dict[str, Any]],
    label_key: str,
    test_size: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record[label_key]].append(dict(record))

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    for label, label_records in grouped.items():
        del label
        rng.shuffle(label_records)
        if len(label_records) == 1:
            train_records.extend(label_records)
            continue

        val_count = max(1, int(round(len(label_records) * test_size)))
        val_count = min(val_count, len(label_records) - 1)
        train_records.extend(label_records[val_count:])
        val_records.extend(label_records[:val_count])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def build_training_arguments(output_dir: str, metric_for_best_model: str, args: argparse.Namespace):
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
        kwargs["save_strategy"] = "epoch"
        kwargs["logging_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
        kwargs["save_strategy"] = "epoch"
        kwargs["logging_strategy"] = "epoch"

    if "save_safetensors" in signature.parameters:
        kwargs["save_safetensors"] = True

    if torch.cuda.is_available() and "fp16" in signature.parameters:
        kwargs["fp16"] = True

    return TrainingArguments(**kwargs)


def save_json(path: str | Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

def save_best_model_pt(
    output_dir: Path,
    task: str,
    model_name: str,
    model,
    metadata: dict,
) -> None:
    payload = {
        "framework": "transformers",
        "task": task,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, output_dir / "best_model.pt")

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_name = args.model_name or choose_default_model(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "sentiment":
        records = normalize_sentiment_examples(load_json(args.data_path))
        train_records, val_records = stratified_split(records, label_key="label", test_size=args.test_size, seed=args.seed)

        label_values = sorted({int(record["label"]) for record in records})
        label2id = {label_value: idx for idx, label_value in enumerate(label_values)}
        id2value = {idx: label_value for label_value, idx in label2id.items()}
        id2label_name = {idx: RESULT_LABEL_NAMES[label_value] for idx, label_value in id2value.items()}
        label_name_to_id = {label_name: idx for idx, label_name in id2label_name.items()}

        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        config = AutoConfig.from_pretrained(model_name, num_labels=len(label_values))
        config.id2label = {idx: label_name for idx, label_name in id2label_name.items()}
        config.label2id = label_name_to_id
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        train_dataset = SentimentTransformerDataset(
            records=train_records,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            use_ocr_hint=args.use_ocr_hint,
            image_dir=args.image_dir,
            ocr_cache_path=args.ocr_cache_path if args.use_ocr_hint else None,
        )
        val_dataset = SentimentTransformerDataset(
            records=val_records,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            use_ocr_hint=args.use_ocr_hint,
            image_dir=args.image_dir,
            ocr_cache_path=args.ocr_cache_path if args.use_ocr_hint else None,
        )

        def compute_metrics(eval_prediction):
            logits = eval_prediction.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            pred_ids = np.asarray(logits).argmax(axis=-1).tolist()
            true_ids = np.asarray(eval_prediction.label_ids).tolist()
            y_pred = [id2value[int(pred_id)] for pred_id in pred_ids]
            y_true = [id2value[int(true_id)] for true_id in true_ids]
            metrics = compute_sentiment_metrics(y_true=y_true, y_pred=y_pred, labels=label_values)
            return {
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }

        trainer = Trainer(
            model=model,
            args=build_training_arguments(str(output_dir), metric_for_best_model="macro_f1", args=args),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        metadata = {
            "framework": "transformers",
            "task": "sentiment",
            "model_name": model_name,
            "max_length": args.max_length,
            "use_ocr_hint": args.use_ocr_hint,
            "id2value": {str(key): value for key, value in id2value.items()},
            "id2label_name": {str(key): value for key, value in id2label_name.items()},
        }

        save_best_model_pt(
            output_dir=output_dir,
            task="sentiment",
            model_name=model_name,
            model=trainer.model,
            metadata=metadata,
        )

    else:
        records = load_json(args.data_path)
        enriched_records: List[Dict[str, Any]] = []
        for record in records:
            item = dict(record)
            item["split_label"] = int(item.get("sentiment", 0))
            enriched_records.append(item)
        train_records, val_records = stratified_split(enriched_records, label_key="split_label", test_size=args.test_size, seed=args.seed)
        for record in train_records:
            record.pop("split_label", None)
        for record in val_records:
            record.pop("split_label", None)

        tag2id = {tag: idx for idx, tag in enumerate(NER_TAGS)}
        id2tag = {idx: tag for tag, idx in tag2id.items()}

        from transformers import (
            AutoConfig,
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        config = AutoConfig.from_pretrained(model_name, num_labels=len(tag2id))
        config.id2label = {idx: tag for idx, tag in id2tag.items()}
        config.label2id = tag2id
        model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

        train_dataset = NERTransformerDataset(train_records, tokenizer=tokenizer, tag2id=tag2id, max_length=args.max_length)
        val_dataset = NERTransformerDataset(val_records, tokenizer=tokenizer, tag2id=tag2id, max_length=args.max_length)

        def compute_metrics(eval_prediction):
            logits = eval_prediction.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            pred_ids = np.asarray(logits).argmax(axis=-1)
            label_ids = np.asarray(eval_prediction.label_ids)
            gold_sequences: List[List[str]] = []
            pred_sequences: List[List[str]] = []
            for pred_row, label_row in zip(pred_ids, label_ids):
                gold_tags: List[str] = []
                pred_tags: List[str] = []
                for pred_id, label_id in zip(pred_row.tolist(), label_row.tolist()):
                    if int(label_id) == -100:
                        continue
                    gold_tags.append(id2tag[int(label_id)])
                    pred_tags.append(id2tag[int(pred_id)])
                gold_sequences.append(gold_tags)
                pred_sequences.append(pred_tags)
            metrics = compute_ner_metrics(true_tag_sequences=gold_sequences, pred_tag_sequences=pred_sequences)
            return {
                "token_accuracy": metrics["token_accuracy"],
                "entity_precision": metrics["entity_precision"],
                "entity_recall": metrics["entity_recall"],
                "entity_f1": metrics["entity_f1"],
            }

        trainer = Trainer(
            model=model,
            args=build_training_arguments(str(output_dir), metric_for_best_model="entity_f1", args=args),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        metadata = {
            "framework": "transformers",
            "task": "ner",
            "model_name": model_name,
            "max_length": args.max_length,
            "id2tag": {str(key): value for key, value in id2tag.items()},
        }

        save_best_model_pt(
            output_dir=output_dir,
            task="ner",
            model_name=model_name,
            model=trainer.model,
            metadata=metadata,
        )

    save_json(output_dir / "enhanced_metadata.json", metadata)
    save_json(output_dir / "eval_metrics.json", {key: float(value) if isinstance(value, np.floating) else value for key, value in eval_metrics.items()})
    save_json(output_dir / "train_split.json", train_records)
    save_json(output_dir / "val_split.json", val_records)

    print(f"Saved model and metadata to: {output_dir}")
    print(json.dumps(eval_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
