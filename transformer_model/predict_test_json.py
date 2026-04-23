from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from transformer_inference import (
    TransformerNERInferencePipeline,
    TransformerSentimentInferencePipeline,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "ner"], required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def normalize_entity(entity: dict):
    return (entity["text"].strip(), entity["label"].strip())


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.task == "sentiment":
        pipeline = TransformerSentimentInferencePipeline(args.checkpoint_dir, device)

        correct = 0
        total = 0
        output = []

        for item in data:
            pred = pipeline.predict(item["text"])

            row = dict(item)
            row["predicted_sentiment"] = pred.value
            row["prediction_confidence"] = pred.confidence
            row["prediction_probabilities"] = pred.probabilities
            output.append(row)

            if "sentiment" in item:
                total += 1
                if int(item["sentiment"]) == int(pred.value):
                    correct += 1

        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        if total > 0:
            print(f"accuracy: {correct / total:.4f}")

        print(f"saved predictions to: {args.output_json}")

    else:
        pipeline = TransformerNERInferencePipeline(args.checkpoint_dir, device)

        tp = 0
        fp = 0
        fn = 0
        output = []

        for item in data:
            pred = pipeline.predict(item["text"])

            row = dict(item)
            row["predicted_entities"] = pred.entities
            output.append(row)

            if "entities" in item:
                gold_set = {normalize_entity(x) for x in item["entities"]}
                pred_set = {normalize_entity(x) for x in pred.entities}

                tp += len(gold_set & pred_set)
                fp += len(pred_set - gold_set)
                fn += len(gold_set - pred_set)

        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        print(f"entity_precision: {precision:.4f}")
        print(f"entity_recall: {recall:.4f}")
        print(f"entity_f1: {f1:.4f}")
        print(f"saved predictions to: {args.output_json}")


if __name__ == "__main__":
    main()