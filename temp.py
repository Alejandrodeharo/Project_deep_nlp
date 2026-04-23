#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def renumerar_match_ids(input_path: str, output_path: str, inicio: int) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El JSON debe contener una lista de partidos.")

    for offset, partido in enumerate(data):
        if not isinstance(partido, dict):
            raise ValueError(f"El elemento en la posición {offset + 1} no es un objeto JSON válido.")
        partido["match_id"] = inicio + offset

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Archivo generado correctamente: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renumera los campos match_id de un JSON en el orden en que aparecen, empezando desde el número que indiques."
    )
    parser.add_argument(
        "input_json",
        help="Ruta del archivo JSON de entrada."
    )
    parser.add_argument(
        "-i",
        "--inicio",
        type=int,
        required=True,
        help="Número desde el que quieres empezar la numeración."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="json_renumerado.json",
        help="Ruta del archivo JSON de salida. Por defecto: json_renumerado.json"
    )

    args = parser.parse_args()
    renumerar_match_ids(args.input_json, args.output, args.inicio)


if __name__ == "__main__":
    main()
