from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


"""
Constantes especiales del vocabulario.

- <PAD>: token usado para rellenar secuencias hasta una longitud común dentro de un batch.
- <UNK>: token usado cuando una palabra no existe en el vocabulario aprendido.
"""
SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1}


"""
Lista fija de etiquetas BIO para la tarea de Named Entity Recognition (NER).

Propósito:
- Definir el espacio completo de etiquetas que el modelo puede predecir.
- Mantener una codificación consistente entre entrenamiento, validación e inferencia.

Formato BIO:
- B-*: inicio de una entidad
- I-*: continuación de una entidad
- O: token fuera de cualquier entidad
- <PAD>: etiqueta de relleno para secuencias con padding
"""
NER_TAGS = [
    "<PAD>",
    "O",
    "B-TEAM", "I-TEAM",
    "B-STADIUM", "I-STADIUM",
    "B-PLAYER", "I-PLAYER",
    "B-COACH", "I-COACH",
]


def set_seed(seed: int) -> None:
    """
    Fija la semilla aleatoria en todos los componentes relevantes para mejorar la reproducibilidad.

    Qué hace:
    - Configura semillas para numpy, random y torch.
    - Si hay GPU disponible, también fija la semilla de CUDA.
    - Activa opciones deterministas en PyTorch/cuDNN para reducir variaciones entre ejecuciones.

    Propósito:
    - Conseguir que los experimentos sean lo más repetibles posible.
    - Facilitar depuración, comparación de modelos y validación de resultados.

    Parámetros:
    - seed: entero usado como semilla global.
    """
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


def save_checkpoint(payload: Dict[str, Any], path: str) -> None:
    """
    Guarda un checkpoint de PyTorch en disco.

    Qué hace:
    - Crea automáticamente la carpeta destino si no existe.
    - Guarda el diccionario `payload` usando `torch.save`.

    Propósito:
    - Persistir el estado del entrenamiento o del modelo para poder reutilizarlo después.
    - Permitir evaluación, inferencia o reanudación del entrenamiento.

    Parámetros:
    - payload: diccionario con la información a guardar
      (por ejemplo: pesos del modelo, configuración, vocabulario, etiquetas, etc.).
    - path: ruta del archivo donde se guardará el checkpoint.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """
    Carga un checkpoint previamente guardado con PyTorch.

    Qué hace:
    - Lee el archivo desde disco con `torch.load`.
    - Permite indicar el dispositivo donde cargarlo (CPU o GPU).

    Propósito:
    - Recuperar un modelo entrenado y su metadata para evaluación o predicción.
    - Evitar tener que reentrenar desde cero.

    Parámetros:
    - path: ruta del archivo del checkpoint.
    - map_location: dispositivo de carga; por defecto se usa CPU.

    Devuelve:
    - Un diccionario con el contenido del checkpoint.
    """
    return torch.load(path, map_location=map_location)


def load_json(path: str) -> List[Dict[str, Any]]:
    """
    Carga un archivo JSON y verifica que su contenido sea una lista de ejemplos.

    Qué hace:
    - Abre el archivo JSON.
    - Lo parsea a objetos Python.
    - Comprueba que el contenido principal sea una lista.

    Propósito:
    - Estandarizar la lectura del dataset.
    - Detectar temprano errores de formato en los datos de entrada.

    Parámetros:
    - path: ruta al archivo JSON.

    Devuelve:
    - Una lista de diccionarios, donde cada diccionario representa un ejemplo del dataset.

    Lanza:
    - ValueError si el JSON no contiene una lista en la raíz.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, but got {type(data).__name__}")
    return data


def tokenize(text: str) -> List[str]:
    """
    Tokeniza un texto en palabras y signos de puntuación.

    Qué hace:
    - Convierte el texto a minúsculas.
    - Usa una expresión regular para separar palabras y puntuación.
    - Mantiene tokens como palabras con guiones o apóstrofes.

    Propósito:
    - Generar la secuencia de tokens para tareas de clasificación de texto.
    - Crear una representación consistente para construir el vocabulario y codificar textos.

    Parámetros:
    - text: texto de entrada.

    Devuelve:
    - Lista de tokens en minúsculas.
    """
    return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text.lower())


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokeniza un texto devolviendo, además de cada token, su posición exacta en el string original.

    Qué hace:
    - Separa palabras y puntuación.
    - Para cada token, devuelve:
      (token, posición_inicial, posición_final)

    Propósito:
    - Poder alinear entidades del texto original con los tokens generados.
    - Es fundamental para construir etiquetas BIO en la tarea NER.

    Parámetros:
    - text: texto original.

    Devuelve:
    - Lista de tuplas (token, start_char, end_char).
    """
    pattern = r"\w+(?:[-']\w+)*|[^\w\s]"
    return [(m.group(), m.start(), m.end()) for m in re.finditer(pattern, text)]


def normalize_sentiment_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normaliza ejemplos de clasificación de sentimiento para que todos usen la misma clave de etiqueta.

    Qué hace:
    - Acepta datasets donde la clase aparece en `label` o en `sentiment`.
    - Si un ejemplo no tiene `label` pero sí `sentiment`, copia su valor a `label`.
    - Verifica que cada ejemplo tenga también el campo `text`.

    Propósito:
    - Unificar formatos de dataset distintos sin cambiar el resto del pipeline.
    - Permitir que el entrenamiento y la evaluación trabajen siempre con una misma estructura.

    Parámetros:
    - examples: lista de ejemplos del dataset.

    Devuelve:
    - Nueva lista de ejemplos normalizados, donde siempre existe la clave `label`.

    Lanza:
    - KeyError si falta tanto `label` como `sentiment`, o si falta `text`.
    """
    normalized: List[Dict[str, Any]] = []

    for ex in examples:
        item = dict(ex)

        if "label" not in item:
            if "sentiment" in item:
                item["label"] = item["sentiment"]
            else:
                raise KeyError("Each sentiment example must contain either 'label' or 'sentiment'.")

        if "text" not in item:
            raise KeyError("Each example must contain 'text'.")

        normalized.append(item)

    return normalized


def build_vocab_from_texts(texts: List[str], max_vocab_size: int = 50000) -> Dict[str, int]:
    """
    Construye un vocabulario a partir de una lista de textos.

    Qué hace:
    - Tokeniza cada texto.
    - Cuenta la frecuencia de cada token.
    - Crea un diccionario token -> índice.
    - Reserva primero los índices de los tokens especiales (`<PAD>` y `<UNK>`).
    - Añade después los tokens más frecuentes hasta `max_vocab_size`.

    Propósito:
    - Transformar palabras en índices numéricos que puedan ser usados por el modelo.
    - Limitar el tamaño del vocabulario para controlar memoria y complejidad.

    Parámetros:
    - texts: lista de textos de entrenamiento.
    - max_vocab_size: número máximo de tokens frecuentes a añadir, además de los especiales.

    Devuelve:
    - Diccionario `word2idx`.
    """
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    word2idx = dict(SPECIAL_TOKENS)
    for word, _ in counter.most_common(max_vocab_size):
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    return word2idx


def build_vocab_from_ner_examples(
    examples: List[Dict[str, Any]],
    max_vocab_size: int = 50000,
) -> Dict[str, int]:
    """
    Construye el vocabulario para NER a partir de los tokens generados por `build_bio_tags`.

    Qué hace:
    - Para cada ejemplo, obtiene los tokens del texto usando la misma lógica del pipeline NER.
    - Cuenta la frecuencia de esos tokens en minúsculas.
    - Construye un diccionario token -> índice.

    Propósito:
    - Mantener coherencia exacta entre el preprocesado usado para crear etiquetas BIO
      y el preprocesado usado para construir el vocabulario.
    - Replicar el comportamiento del código NER funcional original.

    Parámetros:
    - examples: ejemplos del dataset NER.
    - max_vocab_size: número máximo de tokens frecuentes a incluir.

    Devuelve:
    - Diccionario `word2idx`.
    """
    counter = Counter()

    for ex in examples:
        tokens, _ = build_bio_tags(ex)
        counter.update(tok.lower() for tok in tokens)

    word2idx = dict(SPECIAL_TOKENS)
    for word, _ in counter.most_common(max_vocab_size):
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    return word2idx


def encode_text(text: str, word2idx: Dict[str, int]) -> List[int]:
    """
    Convierte un texto en una secuencia de índices usando un vocabulario dado.

    Qué hace:
    - Tokeniza el texto.
    - Reemplaza cada token por su índice en `word2idx`.
    - Si un token no existe en el vocabulario, usa el índice de `<UNK>`.

    Propósito:
    - Transformar texto libre en entrada numérica compatible con redes neuronales.

    Parámetros:
    - text: texto de entrada.
    - word2idx: vocabulario token -> índice.

    Devuelve:
    - Lista de enteros que representa el texto tokenizado.
    """
    return [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokenize(text)]


def find_all_occurrences(text: str, substring: str) -> List[Tuple[int, int]]:
    """
    Encuentra todas las apariciones exactas de un substring dentro de un texto.

    Qué hace:
    - Busca todas las coincidencias exactas de `substring` en `text`.
    - Devuelve las posiciones de inicio y fin de cada coincidencia.

    Propósito:
    - Localizar en el texto original todas las menciones de una entidad.
    - Facilitar la conversión de entidades anotadas a etiquetas BIO por token.

    Parámetros:
    - text: texto completo.
    - substring: fragmento a buscar.

    Devuelve:
    - Lista de tuplas (start_char, end_char).
    """
    matches = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(substring)))
        start = idx + 1
    return matches


def build_bio_tags(example: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Convierte un ejemplo NER en tokens y etiquetas BIO alineadas por token.

    Qué hace:
    - Tokeniza el texto con offsets de caracteres.
    - Inicializa todos los tokens con la etiqueta `O`.
    - Recorre las entidades del ejemplo, priorizando primero las más largas.
    - Busca todas las apariciones exactas del texto de cada entidad.
    - Marca los tokens solapados con etiquetas BIO:
      - `B-<LABEL>` para el primer token de la entidad
      - `I-<LABEL>` para los siguientes tokens
    - Evita sobreescribir tokens ya etiquetados.

    Propósito:
    - Transformar anotaciones a nivel de texto en supervisión a nivel de token.
    - Preparar los datos para entrenar un modelo NER secuencial.

    Parámetros:
    - example: diccionario con al menos `text` y opcionalmente `entities`.

    Devuelve:
    - tokens: lista de tokens del texto
    - tags: lista de etiquetas BIO alineadas con esos tokens

    Nota:
    - Esta función replica la lógica del script NER funcional original.
    """
    text = example["text"]
    entities = example.get("entities", [])

    token_spans = tokenize_with_offsets(text)
    tokens = [tok for tok, _, _ in token_spans]
    tags = ["O"] * len(tokens)

    entities_sorted = sorted(entities, key=lambda e: len(e["text"]), reverse=True)

    for ent in entities_sorted:
        ent_text = ent["text"]
        ent_label = str(ent["label"]).upper()

        occurrences = find_all_occurrences(text, ent_text)

        for ent_start, ent_end in occurrences:
            overlapping_token_ids = []

            for i, (_, tok_start, tok_end) in enumerate(token_spans):
                if tok_start < ent_end and tok_end > ent_start:
                    overlapping_token_ids.append(i)

            if not overlapping_token_ids:
                continue

            if any(tags[idx] != "O" for idx in overlapping_token_ids):
                continue

            tags[overlapping_token_ids[0]] = f"B-{ent_label}"
            for idx in overlapping_token_ids[1:]:
                tags[idx] = f"I-{ent_label}"

    return tokens, tags


def train_test_split_manual(
    data: List[Dict[str, Any]],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Divide manualmente un dataset en entrenamiento y test/validación.

    Qué hace:
    - Crea una copia de los datos.
    - Mezcla los ejemplos de forma aleatoria controlada por una semilla.
    - Separa el dataset en dos partes según `test_size`.

    Propósito:
    - Generar conjuntos de entrenamiento y validación/test sin depender de librerías externas.
    - Mantener reproducibilidad en la partición de datos.

    Parámetros:
    - data: lista de ejemplos.
    - test_size: proporción destinada a validación/test.
    - seed: semilla para mezclar los datos.

    Devuelve:
    - Tupla `(train_data, test_data)`.
    """
    data = data.copy()
    random.Random(seed).shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]


def build_label_mappings(examples: List[Dict[str, Any]]) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    """
    Construye los mapeos entre etiquetas de clasificación e índices numéricos.

    Qué hace:
    - Extrae todas las etiquetas únicas presentes en `examples`.
    - Las ordena.
    - Crea:
      - `label2idx`: etiqueta -> índice
      - `idx2label`: índice -> etiqueta

    Propósito:
    - Convertir etiquetas simbólicas o numéricas en una codificación estable para el modelo.
    - Permitir convertir predicciones numéricas de vuelta a su etiqueta original.

    Parámetros:
    - examples: lista de ejemplos con la clave `label`.

    Devuelve:
    - Tupla `(label2idx, idx2label)`.
    """
    labels = sorted({ex["label"] for ex in examples})
    label2idx = {label: idx for idx, label in enumerate(labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label


def build_tag_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Construye los mapeos entre etiquetas BIO de NER e índices numéricos.

    Qué hace:
    - Usa la lista fija `NER_TAGS`.
    - Crea:
      - `tag2idx`: etiqueta -> índice
      - `idx2tag`: índice -> etiqueta

    Propósito:
    - Garantizar una codificación consistente de las etiquetas NER en todo el pipeline.
    - Permitir pasar de etiquetas textuales a enteros y viceversa.

    Devuelve:
    - Tupla `(tag2idx, idx2tag)`.
    """
    tag2idx = {tag: i for i, tag in enumerate(NER_TAGS)}
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    return tag2idx, idx2tag


class SentimentDataset(Dataset):
    """
    Dataset de PyTorch para clasificación de sentimiento.

    Qué hace:
    - Recibe ejemplos de texto etiquetados.
    - Convierte cada texto en una secuencia de índices.
    - Convierte cada etiqueta en su identificador numérico.
    - Guarda también la longitud de cada secuencia.

    Propósito:
    - Proporcionar una interfaz compatible con `DataLoader`.
    - Facilitar batching, padding y entrenamiento del modelo de clasificación.
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        word2idx: Dict[str, int],
        label2idx: Dict[Any, int],
    ) -> None:
        """
        Inicializa el dataset de sentimiento.

        Parámetros:
        - examples: lista de ejemplos con `text` y `label`.
        - word2idx: vocabulario token -> índice.
        - label2idx: mapeo etiqueta -> índice.
        """
        self.samples: List[Dict[str, Any]] = []

        for ex in examples:
            text = ex["text"]
            label = ex["label"]
            input_ids = encode_text(text, word2idx)

            self.samples.append(
                {
                    "text": text,
                    "label": label,
                    "input_ids": input_ids,
                    "label_id": label2idx[label],
                    "length": len(input_ids),
                }
            )

    def __len__(self) -> int:
        """
        Devuelve el número total de ejemplos del dataset.

        Propósito:
        - Permitir que PyTorch conozca el tamaño del dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Devuelve un ejemplo individual del dataset.

        Propósito:
        - Permitir que `DataLoader` recupere elementos por índice.
        """
        return self.samples[idx]


class NERDataset(Dataset):
    """
    Dataset de PyTorch para Named Entity Recognition (NER).

    Qué hace:
    - Convierte cada ejemplo anotado en:
      - tokens
      - etiquetas BIO
      - ids de entrada
      - ids de etiquetas
      - longitud de la secuencia

    Propósito:
    - Preparar los datos de NER para su uso con `DataLoader` y el modelo secuencial.
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        word2idx: Dict[str, int],
        tag2idx: Dict[str, int],
    ) -> None:
        """
        Inicializa el dataset de NER.

        Parámetros:
        - examples: lista de ejemplos con `text` y opcionalmente `entities`.
        - word2idx: vocabulario token -> índice.
        - tag2idx: mapeo etiqueta BIO -> índice.
        """
        self.samples: List[Dict[str, Any]] = []

        for ex in examples:
            tokens, tags = build_bio_tags(ex)

            input_ids = [word2idx.get(tok.lower(), word2idx["<UNK>"]) for tok in tokens]
            tag_ids = [tag2idx[tag] for tag in tags]

            self.samples.append(
                {
                    "text": ex["text"],
                    "tokens": tokens,
                    "tags": tags,
                    "input_ids": input_ids,
                    "tag_ids": tag_ids,
                    "length": len(tokens),
                }
            )

    def __len__(self) -> int:
        """
        Devuelve el número total de ejemplos del dataset.

        Propósito:
        - Permitir que PyTorch conozca el tamaño del dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Devuelve un ejemplo individual del dataset.

        Propósito:
        - Permitir que `DataLoader` recupere elementos por índice.
        """
        return self.samples[idx]


def sentiment_collate_fn(batch: List[Dict[str, Any]]):
    """
    Función de colación para clasificación de sentimiento.

    Qué hace:
    - Ordena el batch por longitud descendente.
    - Calcula la longitud máxima del batch.
    - Aplica padding a las secuencias para que todas tengan la misma longitud.
    - Devuelve tensores de inputs, labels y longitudes, además del texto original.

    Propósito:
    - Preparar lotes homogéneos para el modelo.
    - Facilitar operaciones de batching en PyTorch.

    Parámetros:
    - batch: lista de ejemplos individuales del dataset.

    Devuelve:
    - inputs: tensor [batch, seq_len]
    - labels: tensor [batch]
    - lengths: tensor [batch]
    - texts: lista de textos originales
    """
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    max_len = max(x["length"] for x in batch) if batch else 0

    inputs = []
    labels = []
    texts = []

    for x in batch:
        pad_len = max_len - x["length"]
        inputs.append(x["input_ids"] + [0] * pad_len)
        labels.append(x["label_id"])
        texts.append(x["text"])

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        lengths,
        texts,
    )


def ner_collate_fn(batch: List[Dict[str, Any]]):
    """
    Función de colación para Named Entity Recognition (NER).

    Qué hace:
    - Ordena el batch por longitud descendente.
    - Calcula la longitud máxima del batch.
    - Aplica padding tanto a los inputs como a las etiquetas por token.
    - Conserva también los tokens y textos originales para inspección o debugging.

    Propósito:
    - Construir batches listos para entrenar un modelo secuencial de NER.
    - Asegurar que entradas y etiquetas tengan dimensiones compatibles dentro de cada lote.

    Parámetros:
    - batch: lista de ejemplos individuales del dataset NER.

    Devuelve:
    - padded_inputs: tensor [batch, seq_len]
    - padded_tags: tensor [batch, seq_len]
    - lengths: tensor [batch]
    - token_lists: lista de listas de tokens originales
    - text_list: lista de textos originales
    """
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    max_len = max(x["length"] for x in batch) if batch else 0

    padded_inputs = []
    padded_tags = []
    token_lists = []
    text_list = []

    for x in batch:
        pad_len = max_len - x["length"]
        padded_inputs.append(x["input_ids"] + [0] * pad_len)
        padded_tags.append(x["tag_ids"] + [0] * pad_len)
        token_lists.append(x["tokens"])
        text_list.append(x["text"])

    return (
        torch.tensor(padded_inputs, dtype=torch.long),
        torch.tensor(padded_tags, dtype=torch.long),
        lengths,
        token_lists,
        text_list,
    )


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calcula la accuracy para una tarea de clasificación.

    Qué hace:
    - Obtiene la clase predicha como el índice con mayor logit.
    - Compara las predicciones con las etiquetas reales.
    - Devuelve la proporción de aciertos.

    Propósito:
    - Medir el rendimiento del modelo en clasificación de texto.

    Parámetros:
    - logits: tensor de salida del modelo con scores por clase.
    - labels: tensor con las etiquetas reales.

    Devuelve:
    - Accuracy en formato float.
    """
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_tag_idx: int = 0) -> float:
    """
    Calcula la accuracy a nivel de token para NER, ignorando el padding.

    Qué hace:
    - Obtiene la etiqueta predicha por token usando argmax.
    - Crea una máscara para excluir posiciones de padding.
    - Calcula la proporción de tokens correctamente etiquetados sobre los tokens válidos.

    Propósito:
    - Medir el rendimiento real del modelo NER sin que el padding distorsione la métrica.

    Parámetros:
    - logits: tensor de salida del modelo con scores por etiqueta y token.
    - labels: tensor con etiquetas reales por token.
    - pad_tag_idx: índice de la etiqueta de padding.

    Devuelve:
    - Accuracy por token en formato float.
    """
    preds = logits.argmax(dim=-1)
    mask = labels != pad_tag_idx
    correct = ((preds == labels) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0