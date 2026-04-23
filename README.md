NLP Football Pipeline

Este repositorio implementa un pipeline híbrido para procesar crónicas de partidos de fútbol y generar alertas cortas en inglés a partir de:

- texto del partido
- entidades extraídas del texto
- resultado inferido del partido
- opcionalmente, una imagen con el marcador

--------------------------------------------------

ARQUITECTURA

El sistema combina cuatro bloques:

1. Modelo de Sentiment / resultado del partido
   Entrada: texto del partido
   Salida: 1, 0 o -1
   Interpretación:
     1 → HOME_WIN
     0 → DRAW
     -1 → AWAY_WIN

2. Modelo de NER (Named Entity Recognition)
   Entrada: texto
   Salida: etiquetas BIO y entidades
   Entidades:
     TEAM
     STADIUM
     PLAYER
     COACH

3. Modelo generativo de alertas
   Modelo: Qwen/Qwen2.5-1.5B-Instruct
   Entrada: sentimiento + entidades + contexto
   Salida: alerta corta en inglés

4. OCR opcional
   Script: image_captioning.py
   Librería: EasyOCR
   Extrae marcador desde imágenes

--------------------------------------------------

ARCHIVOS PRINCIPALES

train.py
Entrena modelos de:
- sentiment
- ner

NER usa una loss estructurada que penaliza:
- omitir entidades
- falsos positivos
- errores BIO

evaluate.py
Evalúa modelos y genera:
- métricas globales
- predicciones por sample

main.py
Pipeline completo:
1. carga modelos
2. OCR opcional
3. sentiment
4. NER
5. generación de alerta
6. guarda JSON final

alert_generation.py
Genera alertas desde datos estructurados

image_captioning.py
Hace OCR de marcador (no captioning real)

train_functions.py
Funciones de entrenamiento y métricas

models.py
Arquitecturas:
- sentiment (MeanPool, CNN, BiLSTM)
- NER (BiLSTM)

utils.py
Utilidades:
- tokenización
- vocabulario
- datasets
- checkpoints

--------------------------------------------------

ARQUITECTURA REAL

No es end-to-end.

Hay módulos independientes:
- Sentiment
- NER
- LLM (Qwen)
- OCR

Se conectan secuencialmente en main.py.

No comparten pesos.

--------------------------------------------------

ENTRENAMIENTO

Sentiment:
- CrossEntropyLoss
- métrica: accuracy

NER:
- BiLSTM
- loss estructurada:
  - CrossEntropy ponderada
  - penalización por errores de entidad
  - penalización BIO

Selección de modelo:
- por entity_f1 (no token_accuracy)

--------------------------------------------------

MÉTRICAS

Sentiment:
- loss
- accuracy

NER:
- loss
- token_accuracy
- entity_precision
- entity_recall
- entity_f1 (principal)
- sample_exact_match

--------------------------------------------------

EVALUACIÓN

evaluate.py devuelve:

Sentiment:
- text
- gold_label
- predicted_label
- probabilities
- correct

NER:
- text
- tokens
- gold_tags
- predicted_tags
- gold_entities
- predicted_entities
- token_accuracy_sample

--------------------------------------------------

GENERACIÓN DE ALERTAS

Modelo: Qwen/Qwen2.5-1.5B-Instruct

Parámetros:
- max_new_tokens=72
- temperature=0.7
- top_p=0.9

No usa GPU por defecto → puede ser lento

--------------------------------------------------

OCR

Lee marcador desde imagen:

Formato:
2 - 1

Usa múltiples regiones y preprocessings:
- grayscale
- threshold
- invert
- CLAHE

--------------------------------------------------

PIPELINE

Flujo:

texto
 → sentiment
 → NER
 → OCR (opcional)
 → contexto
 → generador
 → alerta final

--------------------------------------------------

MODOS DE EJECUCIÓN

predicted
gold_sentiment
gold_ner
structured_only
sentiment_guided

--------------------------------------------------

FORMATO DE DATOS

Sentiment:
{
  "text": "...",
  "sentiment": 1
}

NER:
{
  "text": "...",
  "entities": [
    {"text": "...", "label": "TEAM"}
  ]
}

Pipeline:
{
  "match_id": 1001,
  "home_team": "...",
  "away_team": "...",
  "text": "..."
}

--------------------------------------------------

INSTALACIÓN

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

--------------------------------------------------

COMANDOS

Entrenar Sentiment:
python train.py --task sentiment ...

Entrenar NER:
python train.py --task ner ...

Evaluar:
python evaluate.py ...

Pipeline:
python main.py ...

Alertas:
python alert_generation.py ...

OCR:
python image_captioning.py

--------------------------------------------------

FLUJO RECOMENDADO

1. entrenar sentiment
2. entrenar NER
3. evaluar ambos
4. ejecutar pipeline

--------------------------------------------------

DETALLES IMPORTANTES

- Qwen no se entrena aquí
- cada modelo tiene su vocabulario
- NER usa BIO fijo
- no hay CRF
- OCR modifica el texto
- image_captioning es OCR

--------------------------------------------------

USO

1. Entrenar Sentiment:
python train.py \
  --task sentiment \
  --data_path train.json \
  --model_name bilstm \
  --epochs 15 \
  --batch_size 8 \
  --save_path models/sentiment_bilstm.pt


2. Entrenar NER:
python train.py \
  --task ner \
  --data_path train.json \
  --model_name bilstm \
  --epochs 15 \
  --batch_size 8 \
  --save_path models/ner_bilstm_structured.pt \
  --ner_lambda_miss 1.2 \
  --ner_lambda_fp 1.0 \
  --ner_lambda_transition 0.8 \
  --ner_weight_o 1.0 \
  --ner_weight_b 2.0 \
  --ner_weight_i 2.5

3. Evaluar Sentiment:
python evaluate.py \
  --checkpoint models/sentiment_bilstm.pt \
  --data_path test.json \
  --output_path outputs/sentiment.json

4. Evaluar NER:
python evaluate.py \
  --checkpoint models/ner_bilstm_structured.pt \
  --data_path test.json \
  --output_path outputs/ner.json

5. Ejecutar Pipeline Completo:
python main.py \
  --data-path test.json \
  --sentiment-checkpoint models/sentiment_bilstm.pt \
  --ner-checkpoint models/ner_bilstm_structured.pt \
  --mode predicted \
  --output-path outputs/pipeline.json