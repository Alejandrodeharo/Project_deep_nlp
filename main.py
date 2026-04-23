from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from alert_generation import NeuralAlertGenerator, build_ner_summary, sentiment_to_label
from train_functions import build_model
from utils import encode_text, load_checkpoint, load_json, tokenize_with_offsets

import easyocr
from image_captioning import read_score


DEFAULT_OUTPUT_PATH = "./pipeline_outputs.json"
DEFAULT_GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PIPELINE_MODES = (
    "predicted",
    "gold_sentiment",
    "gold_ner",
    "structured_only",
    "sentiment_guided",
)


@dataclass
class SentimentPrediction:
    value: int
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class NERPrediction:
    entities: List[Dict[str, str]]
    tags: List[str]
    tokens: List[str]


class SentimentInferencePipeline:
    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        if checkpoint.get("task") != "sentiment":
            raise ValueError(
                f"Checkpoint {checkpoint_path} is for task={checkpoint.get('task')}, expected sentiment."
            )

        metadata = checkpoint["metadata"]
        config = checkpoint["config"]

        model, _ = build_model(
            task="sentiment",
            model_name=checkpoint["model_name"],
            vocab_size=len(metadata["word2idx"]),
            num_outputs=metadata["num_outputs"],
            embed_dim=config.get("embed_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.to(device)
        self.model.eval()

        self.device = device
        self.word2idx = metadata["word2idx"]
        self.idx2label = checkpoint["metadata"]["idx2label"]
        self.unk_idx = self.word2idx["<UNK>"]

    def predict(self, text: str) -> SentimentPrediction:
        input_ids = encode_text(text, self.word2idx)
        if not input_ids:
            input_ids = [self.unk_idx]

        inputs = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(input_ids)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(inputs, lengths)
            probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = int(probs.argmax().item())
        label = int(self.idx2label[pred_idx])
        probabilities = {
            str(self.idx2label[idx]): float(prob.item())
            for idx, prob in enumerate(probs)
        }

        return SentimentPrediction(
            value=label,
            confidence=float(probs[pred_idx].item()),
            probabilities=probabilities,
        )


class NERInferencePipeline:
    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        if checkpoint.get("task") != "ner":
            raise ValueError(
                f"Checkpoint {checkpoint_path} is for task={checkpoint.get('task')}, expected ner."
            )

        metadata = checkpoint["metadata"]
        config = checkpoint["config"]

        model, _ = build_model(
            task="ner",
            model_name=checkpoint["model_name"],
            vocab_size=len(metadata["word2idx"]),
            num_outputs=metadata["num_outputs"],
            embed_dim=config.get("embed_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.to(device)
        self.model.eval()

        self.device = device
        self.word2idx = metadata["word2idx"]
        self.idx2tag = checkpoint["metadata"]["idx2tag"]
        self.unk_idx = self.word2idx["<UNK>"]

    def predict(self, text: str) -> NERPrediction:
        token_spans = tokenize_with_offsets(text)
        if not token_spans:
            return NERPrediction(entities=[], tags=[], tokens=[])

        tokens = [token for token, _, _ in token_spans]
        input_ids = [self.word2idx.get(token.lower(), self.unk_idx) for token in tokens]

        inputs = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(input_ids)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(inputs, lengths)
            pred_ids = logits.argmax(dim=-1)[0][: len(tokens)].tolist()

        tags = [self.idx2tag[int(tag_id)] for tag_id in pred_ids]
        entities = decode_bio_predictions(text=text, token_spans=token_spans, tags=tags)

        return NERPrediction(entities=entities, tags=tags, tokens=tokens)


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


def normalize_entities_for_compare(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized = []
    seen = set()

    for entity in entities:
        text = str(entity["text"]).strip()
        label = str(entity["label"]).strip().upper()
        key = (text, label)
        if key not in seen:
            normalized.append({"text": text, "label": label})
            seen.add(key)

    normalized.sort(key=lambda x: (x["label"], x["text"]))
    return normalized


def validate_record(record: Dict[str, Any], index: int) -> None:
    required_fields = ["text", "home_team", "away_team"]
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        raise ValueError(f"Record {index} is missing required fields: {missing_fields}")


def build_sentiment_guided_context(
    record: Dict[str, Any],
    sentiment_value: int,
    entities: List[Dict[str, str]],
) -> str:
    home_team = record["home_team"]
    away_team = record["away_team"]

    if sentiment_value == 1:
        outcome_hint = f"The predicted result is a home win for {home_team}."
    elif sentiment_value == -1:
        outcome_hint = f"The predicted result is an away win for {away_team}."
    else:
        outcome_hint = f"The predicted result is a draw between {home_team} and {away_team}."

    entity_hint = build_ner_summary(entities).replace("\n", " | ")
    return (
        f"{outcome_hint} "
        f"Use this signal to keep the alert outcome consistent. "
        f"Named entities detected: {entity_hint}. "
        f"Original match report: {record['text']}"
    )


def resolve_sentiment(
    record: Dict[str, Any],
    prediction: SentimentPrediction,
    mode: str,
) -> Tuple[int, str]:
    if mode == "gold_sentiment":
        if "sentiment" not in record:
            raise ValueError("Mode 'gold_sentiment' requires a 'sentiment' field in the input records.")
        return int(record["sentiment"]), "gold"

    return prediction.value, "predicted"


def resolve_entities(
    record: Dict[str, Any],
    prediction: NERPrediction,
    mode: str,
) -> Tuple[List[Dict[str, str]], str]:
    if mode == "gold_ner":
        if "entities" not in record:
            raise ValueError("Mode 'gold_ner' requires an 'entities' field in the input records.")
        entities = record["entities"]
        if not isinstance(entities, list):
            raise ValueError("Field 'entities' must be a list.")
        return entities, "gold"

    return prediction.entities, "predicted"


def build_generation_context(
    record: Dict[str, Any],
    mode: str,
    used_sentiment: int,
    used_entities: List[Dict[str, str]],
) -> Optional[str]:
    if mode == "structured_only":
        return None
    if mode == "sentiment_guided":
        return build_sentiment_guided_context(record, used_sentiment, used_entities)
    return record["text"]


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_pipeline_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_records": len(records),
        "sentiment_metrics": None,
        "ner_metrics": None,
    }

    sentiment_gold = 0
    sentiment_correct = 0

    ner_gold = 0
    ner_exact_match = 0

    for record in records:
        pipeline_outputs = record.get("pipeline_outputs", {})

        if "sentiment" in record:
            sentiment_gold += 1
            if int(record["sentiment"]) == int(pipeline_outputs["predicted_sentiment"]):
                sentiment_correct += 1

        if "entities" in record and isinstance(record["entities"], list):
            ner_gold += 1

            gold_entities = normalize_entities_for_compare(record["entities"])
            pred_entities = normalize_entities_for_compare(pipeline_outputs["predicted_entities"])

            if gold_entities == pred_entities:
                ner_exact_match += 1

    if sentiment_gold > 0:
        summary["sentiment_metrics"] = {
            "num_samples_with_gold": sentiment_gold,
            "accuracy": sentiment_correct / sentiment_gold,
        }

    if ner_gold > 0:
        summary["ner_metrics"] = {
            "num_samples_with_gold": ner_gold,
            "entity_exact_match": ner_exact_match / ner_gold,
        }

    return summary


def run_pipeline(
    records: Sequence[Dict[str, Any]],
    sentiment_pipeline: SentimentInferencePipeline,
    ner_pipeline: NERInferencePipeline,
    generator: NeuralAlertGenerator,
    mode: str,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    enriched_records: List[Dict[str, Any]] = []

    use_gpu = torch.cuda.is_available()
    ocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    data_dir = Path("data")

    for index, record in enumerate(records):
        validate_record(record, index)
        record = dict(record)

        match_id = record.get("match_id")
        image_score = None

        if match_id is not None:
            for ext in [".png", ".jpg", ".jpeg"]:
                ruta_img = data_dir / f"{match_id}{ext}"
                if ruta_img.exists():
                    try:
                        score = read_score(str(ruta_img), verbose=False, reader=ocr_reader)
                        image_score = score
                        record["text"] += f" The scoreboard image shows a score of {score}."
                        print(f"Imagen {ruta_img.name} procesada: {score}")
                    except Exception as e:
                        print(f"Error leyendo {ruta_img.name}: {e}")
                    break

        sentiment_prediction = sentiment_pipeline.predict(record["text"])
        ner_prediction = ner_pipeline.predict(record["text"])

        used_sentiment, sentiment_source = resolve_sentiment(
            record=record,
            prediction=sentiment_prediction,
            mode=mode,
        )
        used_entities, entities_source = resolve_entities(
            record=record,
            prediction=ner_prediction,
            mode=mode,
        )
        generation_context = build_generation_context(
            record=record,
            mode=mode,
            used_sentiment=used_sentiment,
            used_entities=used_entities,
        )

        generated_alert = generator.generate_alert(
            home_team=record["home_team"],
            away_team=record["away_team"],
            sentiment=used_sentiment,
            entities=used_entities,
            original_text=generation_context,
            max_new_tokens=max_new_tokens,
        )

        output_record = dict(record)
        output_record["generated_alert"] = generated_alert
        output_record["pipeline_mode"] = mode
        output_record["pipeline_outputs"] = {
            "image_score": image_score,
            "predicted_sentiment": sentiment_prediction.value,
            "predicted_sentiment_label": sentiment_to_label(sentiment_prediction.value),
            "predicted_sentiment_confidence": sentiment_prediction.confidence,
            "predicted_sentiment_probabilities": sentiment_prediction.probabilities,
            "predicted_tokens": ner_prediction.tokens,
            "predicted_entities": ner_prediction.entities,
            "predicted_ner_tags": ner_prediction.tags,
            "used_sentiment": used_sentiment,
            "used_sentiment_label": sentiment_to_label(used_sentiment),
            "used_sentiment_source": sentiment_source,
            "used_entities": used_entities,
            "used_entities_source": entities_source,
        }

        if "sentiment" in record:
            output_record["pipeline_outputs"]["sentiment_correct"] = (
                int(record["sentiment"]) == int(sentiment_prediction.value)
            )

        if "entities" in record and isinstance(record["entities"], list):
            output_record["pipeline_outputs"]["ner_exact_match"] = (
                normalize_entities_for_compare(record["entities"]) ==
                normalize_entities_for_compare(ner_prediction.entities)
            )

        enriched_records.append(output_record)

        print(
            f"[{index + 1}/{len(records)}] "
            f"{record['home_team']} vs {record['away_team']} | "
            f"mode={mode} | sentiment={sentiment_to_label(used_sentiment)} | "
            f"entities={len(used_entities)}"
        )

    return enriched_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end inference pipeline: sentiment checkpoint + NER checkpoint + alert generation."
        )
    )
    parser.add_argument("--data-path", required=True, help="Path to the JSON dataset to process.")
    parser.add_argument(
        "--sentiment-checkpoint",
        required=True,
        help="Checkpoint trained for the sentiment task.",
    )
    parser.add_argument(
        "--ner-checkpoint",
        required=True,
        help="Checkpoint trained for the NER task.",
    )
    parser.add_argument(
        "--generator-model-name",
        default=DEFAULT_GENERATION_MODEL,
        help="Hugging Face model used by alert_generation.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the JSON output of the full pipeline will be saved.",
    )
    parser.add_argument(
        "--mode",
        choices=PIPELINE_MODES,
        default="predicted",
        help=(
            "Execution mode. 'predicted' uses both model outputs, 'gold_sentiment' and 'gold_ner' "
            "replace one branch with dataset annotations, 'structured_only' removes the raw text, "
            "and 'sentiment_guided' injects an explicit outcome hint into the generation context."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for the SA and NER checkpoints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records to process.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=72,
        help="Maximum number of generated tokens per alert.",
    )
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)

    records = load_json(args.data_path)
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Loaded {len(records)} records from {args.data_path}")
    print(f"Pipeline mode: {args.mode}")
    print(f"Inference device: {device}")

    sentiment_pipeline = SentimentInferencePipeline(args.sentiment_checkpoint, device)
    ner_pipeline = NERInferencePipeline(args.ner_checkpoint, device)
    generator = NeuralAlertGenerator(model_name=args.generator_model_name)

    outputs = run_pipeline(
        records=records,
        sentiment_pipeline=sentiment_pipeline,
        ner_pipeline=ner_pipeline,
        generator=generator,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
    )

    final_output = {
        "config": {
            "data_path": args.data_path,
            "mode": args.mode,
            "device": str(device),
            "sentiment_checkpoint": args.sentiment_checkpoint,
            "ner_checkpoint": args.ner_checkpoint,
            "generator_model_name": args.generator_model_name,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": build_pipeline_summary(outputs),
        "predictions": outputs,
    }

    save_json(args.output_path, final_output)
    print(f"Saved pipeline outputs to: {args.output_path}")


if __name__ == "__main__":
    main()