import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SOURCE_LENGTH = 2048
DEFAULT_OUTPUT_PATH = "./generated_alerts.json"


def sentiment_to_label(sentiment: int) -> str:
    mapping = {
        1: "HOME_WIN",
        0: "DRAW",
        -1: "AWAY_WIN",
    }
    if sentiment not in mapping:
        raise ValueError(f"Invalid sentiment value: {sentiment}")
    return mapping[sentiment]


def group_entities(entities: List[Dict[str, str]]) -> Dict[str, List[str]]:
    grouped = {"TEAM": [], "STADIUM": [], "PLAYER": [], "COACH": []}
    for entity in entities:
        label = entity["label"]
        text = entity["text"]
        if label in grouped and text not in grouped[label]:
            grouped[label].append(text)
    return grouped


def join_names(names: List[str]) -> str:
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def build_match_facts(
    home_team: str,
    away_team: str,
    sentiment: int,
    entities: List[Dict[str, str]],
    original_text: Optional[str] = None,
) -> str:
    grouped = group_entities(entities)
    stadium = grouped["STADIUM"][0] if grouped["STADIUM"] else "unknown"

    parts = [
        f"result={sentiment_to_label(sentiment)}",
        f"home_team={home_team}",
        f"away_team={away_team}",
        f"stadium={stadium}",
        f"coaches={join_names(grouped['COACH'])}",
        f"players={join_names(grouped['PLAYER'])}",
        f"teams_mentioned={join_names(grouped['TEAM'])}",
    ]

    if original_text:
        parts.append(f"match_report={original_text}")

    return " | ".join(parts)


def build_ner_summary(entities: List[Dict[str, str]]) -> str:
    grouped = group_entities(entities)
    lines = [
        f"TEAM: {join_names(grouped['TEAM'])}",
        f"STADIUM: {join_names(grouped['STADIUM'])}",
        f"PLAYER: {join_names(grouped['PLAYER'])}",
        f"COACH: {join_names(grouped['COACH'])}",
    ]
    return "\n".join(lines)


FEW_SHOT_EXAMPLES = [
    {
        "sa_output": "sentiment=1 -> HOME_WIN",
        "ner_output": (
            "TEAM: SC Freiburg and Sporting CP\n"
            "STADIUM: Europa-Park Stadion\n"
            "PLAYER: Marcus Edwards, Maximilian Eggestein, Morten Hjulmand, and Ritsu Doan\n"
            "COACH: Ruben Amorim"
        ),
        "match_report": (
            "SC Freiburg and Sporting CP played an intense match at Europa-Park Stadion. "
            "Maximilian Eggestein opened the scoring in the first half after Freiburg recovered the ball high up the pitch. "
            "Sporting CP responded through Morten Hjulmand after the break, but Freiburg regained control as Ritsu Doan restored the lead in the closing stages. "
            "Marcus Edwards was booked during a tense second half, and Ruben Amorim pushed Sporting higher late on, yet Freiburg stayed compact and secured a 2-1 home victory."
        ),
        "output": (
            "SC Freiburg beat Sporting CP 2-1 at Europa-Park Stadion, with goalscoring contributions from Maximilian Eggestein and Ritsu Doan proving decisive."
        ),
    },
    {
        "sa_output": "sentiment=-1 -> AWAY_WIN",
        "ner_output": (
            "TEAM: Borussia Dortmund and Newcastle United\n"
            "STADIUM: Signal Iduna Park\n"
            "PLAYER: Niclas Fullkrug, Julian Brandt, Anthony Gordon, Kieran Trippier, and Bruno Guimaraes\n"
            "COACH: none"
        ),
        "match_report": (
            "Borussia Dortmund hosted Newcastle United at Signal Iduna Park in a high-tempo contest. "
            "Julian Brandt gave Dortmund an early lift, but Anthony Gordon brought Newcastle level soon after. "
            "The visitors improved as the game opened up and completed the turnaround when Kieran Trippier finished a well-worked move in the second half. "
            "Bruno Guimaraes helped Newcastle manage the closing stages, while Dortmund struggled to break through despite late pressure."
        ),
        "output": (
            "Newcastle United came from behind to earn a 2-1 away win over Borussia Dortmund at Signal Iduna Park, with Anthony Gordon and Kieran Trippier central to the turnaround."
        ),
    },
]


def build_generation_prompt(
    home_team: str,
    away_team: str,
    sentiment: int,
    entities: List[Dict[str, str]],
    original_text: Optional[str] = None,
) -> str:
    prompt_sections = [
        (
            "You are writing football alerts from upstream NLP outputs."
        ),
        (
            "Task: Generate exactly one short, natural alert in English from the provided Sentiment Analysis output, "
            "NER output, and match report."
        ),
        (
            "Rules:\n"
            "- Make the alert sound like a real sports notification.\n"
            "- State the result clearly, the score is not mandatory just the result of the match, win, draw, defeat.\n"
            "- Use the match report as the main source of context.\n"
            "- Use the NER output to recover the most relevant names, stadium, and teams.\n"
            "- Do not mention the labels 'sentiment' or 'NER' in the alert.\n"
            "- Avoid robotic wording and avoid repeating the same structure every time.\n"
            "- Output only the final alert sentence."
        )
    ]

    for example in FEW_SHOT_EXAMPLES:
        prompt_sections.append("Example input:")
        prompt_sections.append(f"Sentiment Analysis output:\n{example['sa_output']}")
        prompt_sections.append(f"NER output:\n{example['ner_output']}")
        prompt_sections.append(f"match_report:\n{example['match_report']}")
        prompt_sections.append(f"Example output: {example['output']}")

    current_sa_output = f"sentiment={sentiment} -> {sentiment_to_label(sentiment)}"
    current_ner_output = build_ner_summary(entities)
    match_report = original_text if original_text else build_match_facts(
        home_team, away_team, sentiment, entities, original_text=None
    )

    prompt_sections.append(
        "Now write the alert for this match."
    )
    prompt_sections.append(f"Sentiment Analysis output:\n{current_sa_output}")
    prompt_sections.append(f"NER output:\n{current_ner_output}")
    prompt_sections.append(f"match_report:\n{match_report}")
    prompt_sections.append("Alert:")

    return "\n\n".join(prompt_sections)


def validate_record(record: Dict[str, Any], index: int) -> None:
    required_fields = ["home_team", "away_team", "sentiment", "entities", "text"]
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        raise ValueError(f"Record {index} is missing required fields: {missing_fields}")

    if not isinstance(record["entities"], list):
        raise ValueError(f"Record {index} field 'entities' must be a list")


class NeuralAlertGenerator:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_alert(
        self,
        home_team: str,
        away_team: str,
        sentiment: int,
        entities: List[Dict[str, str]],
        original_text: Optional[str] = None,
        max_new_tokens: int = 72,
    ) -> str:
        prompt = build_generation_prompt(
            home_team=home_team,
            away_team=away_team,
            sentiment=sentiment,
            entities=entities,
            original_text=original_text,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise football news writer who converts structured NLP outputs into natural alerts."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
            padding=True,
        )

        device = self.model.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = generated[0][prompt_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("The JSON dataset must be a list of records.")

    return data


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def generate_alerts_for_dataset(
    generator: NeuralAlertGenerator,
    records: List[Dict[str, Any]],
    use_original_text: bool = True,
    max_new_tokens: int = 72,
) -> List[Dict[str, Any]]:
    generated_records: List[Dict[str, Any]] = []

    for index, record in enumerate(records):
        validate_record(record, index)
        generated_alert = generator.generate_alert(
            home_team=record["home_team"],
            away_team=record["away_team"],
            sentiment=record["sentiment"],
            entities=record["entities"],
            original_text=record["text"] if use_original_text else None,
            max_new_tokens=max_new_tokens,
        )

        enriched_record = dict(record)
        enriched_record["generated_alert"] = generated_alert
        generated_records.append(enriched_record)

        print(
            f"[{index + 1}/{len(records)}] "
            f"{record['home_team']} vs {record['away_team']} -> {generated_alert}"
        )

    return generated_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate football alerts from a JSON dataset using a Hugging Face instruction model "
            "and a few-shot prompt. No alert_text field is required."
        )
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the input JSON dataset.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Hugging Face model used for generation.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the JSON with generated alerts will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records to process from the dataset.",
    )
    parser.add_argument(
        "--no-original-text",
        action="store_true",
        help="Exclude the full match report from the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=72,
        help="Maximum number of generated tokens for each alert.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data_path = Path(args.data_path)
    records = load_json(str(data_path))
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Loaded {len(records)} records from {data_path}")
    print("Mode: direct generation from prompt with few-shot examples")

    generator = NeuralAlertGenerator(model_name=args.model_name)
    generated_records = generate_alerts_for_dataset(
        generator=generator,
        records=records,
        use_original_text=not args.no_original_text,
        max_new_tokens=args.max_new_tokens,
    )

    save_json(args.output_path, generated_records)
    print(f"Saved generated alerts to: {args.output_path}")


if __name__ == "__main__":
    main()