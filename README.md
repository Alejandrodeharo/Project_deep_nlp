# NLP Football Pipeline

Este repositorio implementa un pipeline híbrido para generar alertas de partidos de fútbol a partir de texto y, opcionalmente, imágenes del marcador.

La arquitectura combina **tres bloques distintos**:

1. **Modelo de Sentiment / resultado del partido**
   - Entrada: texto del partido.
   - Salida: `1`, `0` o `-1`.
   - Interpretación:
     - `1` → `HOME_WIN`
     - `0` → `DRAW`
     - `-1` → `AWAY_WIN`

2. **Modelo de NER (Named Entity Recognition)**
   - Entrada: texto del partido.
   - Salida: etiquetas BIO por token.
   - Entidades soportadas:
     - `TEAM`
     - `STADIUM`
     - `PLAYER`
     - `COACH`

3. **Modelo generativo de alertas**
   - Modelo Hugging Face: `Qwen/Qwen2.5-1.5B-Instruct` por defecto.
   - Entrada: resultado, entidades y contexto textual.
   - Salida: una alerta corta en inglés, con estilo de notificación deportiva.

Además, existe un módulo OCR opcional:

4. **Lector OCR del marcador desde imagen**
   - Script: `image_captioning.py`
   - Librería principal: `EasyOCR`
   - Extrae un marcador visible de imágenes en `data/`.
   - `main.py` puede integrar ese marcador en el texto antes de inferir.

---

# 1. Qué hace cada archivo

## `train.py`
Entrena un checkpoint para **una sola tarea**:
- `sentiment`
- `ner`

No entrena el modelo generativo. El generador se usa ya preentrenado desde Hugging Face.

## `evaluate.py`
Carga un checkpoint guardado y evalúa ese modelo sobre un dataset JSON.

## `main.py`
Es el **pipeline completo de inferencia**. Hace, en este orden:

1. Carga el checkpoint de sentimiento.
2. Carga el checkpoint de NER.
3. Carga el generador `Qwen/Qwen2.5-1.5B-Instruct`.
4. Si existe una imagen asociada al `match_id`, intenta leer el marcador con OCR.
5. Añade ese marcador al texto.
6. Ejecuta sentimiento sobre el texto.
7. Ejecuta NER sobre el texto.
8. Decide si usar salidas predichas o anotaciones gold según el `--mode`.
9. Construye un prompt few-shot.
10. Genera una alerta final.
11. Guarda un JSON enriquecido con todas las salidas intermedias.

## `alert_generation.py`
Genera alertas a partir de:
- sentimiento
- entidades NER
- texto original o contexto estructurado

Este script también puede ejecutarse **por separado**, sin `main.py`, siempre que el JSON ya tenga:
- `home_team`
- `away_team`
- `sentiment`
- `entities`
- `text`

## `image_captioning.py`
Lee imágenes de marcador con OCR y devuelve resultados como:

```text
101.png: 2 - 1
```

## `train_functions.py`
Contiene la lógica reutilizable de:
- construcción de dataloaders
- construcción de modelos
- train step
- validation step
- test step

## `models.py`
Define las arquitecturas.

Para sentimiento:
- `MeanPoolClassifier`
- `CNNTextClassifier`
- `BiLSTMClassifier`

Para NER:
- `BiLSTMNER`

## `utils.py`
Contiene utilidades clave:
- tokenización
- vocabulario
- BIO tags
- datasets PyTorch
- collate functions
- guardado/carga de checkpoints
- seed y reproducibilidad

---

# 2. Arquitectura real del sistema

## 2.1 No es un único modelo
Este repositorio **no usa un solo modelo end-to-end**. Usa varios bloques encadenados:

- **Modelo 1:** clasificador de sentimiento/resultado
- **Modelo 2:** modelo NER
- **Modelo 3:** LLM generativo (`Qwen/Qwen2.5-1.5B-Instruct`)
- **Módulo adicional opcional:** OCR con EasyOCR

## 2.2 Cómo se “sincronizan”
La sincronización aquí no es concurrente ni distribuida. Es simplemente **secuencial y por paso de datos**.

El flujo real en `main.py` es:

```text
JSON record
   └─> texto del partido
        ├─> SentimentInferencePipeline.predict(text)
        ├─> NERInferencePipeline.predict(text)
        └─> (opcional) OCR score desde imagen asociada al match_id

salidas estructuradas
   └─> generator.generate_alert(...)
        └─> alerta final
```

### Importante
- **Sentiment y NER no se llaman entre sí.** Ambos leen el mismo texto.
- **No comparten pesos.** Son modelos independientes.
- **La unión ocurre en `main.py`.** Ese archivo recoge las dos predicciones y las entrega al generador.
- **El generador tampoco retroalimenta a los modelos anteriores.** Solo consume sus salidas.

---

# 3. Explicación detallada de la ejecución

# 3.1 Entrenamiento (`train.py`)

## Paso 1: parseo de argumentos
`train.py` recibe parámetros como:
- `--task`
- `--data_path`
- `--model_name`
- `--epochs`
- `--batch_size`
- `--save_path`

## Paso 2: selección de dispositivo
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Si hay GPU CUDA, entrena en GPU. Si no, en CPU.

## Paso 3: reproducibilidad
```python
set_seed(args.seed)
```
Fija semilla para `random`, `numpy` y `torch`.

## Paso 4: construcción de datos
`build_dataloaders(...)` hace varias cosas:

### Si `task == "sentiment"`
- Normaliza ejemplos para asegurar que haya `label`.
- Parte el dataset en train/val.
- Construye vocabulario desde los textos de train.
- Construye `label2idx` e `idx2label`.
- Crea `SentimentDataset`.
- Aplica `sentiment_collate_fn` para padding.

### Si `task == "ner"`
- Parte el dataset en train/val.
- Construye vocabulario desde los tokens BIO de train.
- Construye `tag2idx` e `idx2tag`.
- Crea `NERDataset`.
- Aplica `ner_collate_fn` para padding de inputs y tags.

## Paso 5: construcción del modelo
`build_model(...)` usa `task` y `model_name`.

### Para sentimiento
Opciones reales:
- `meanpool`
- `cnn`
- `bilstm`

### Para NER
Siempre construye:
- `bilstm_ner`

Aunque pases `--model_name bilstm` en NER, internamente el modelo real guardado será `bilstm_ner`.

## Paso 6: criterio y optimizador
- Sentiment: `CrossEntropyLoss()`
- NER: `CrossEntropyLoss(ignore_index=<PAD>)`
- Optimizador: `Adam`
- Scheduler: `ReduceLROnPlateau`

## Paso 7: loop de entrenamiento
Cada época:
1. `train_step(...)`
2. `val_step(...)`
3. si mejora la métrica de validación, guarda checkpoint
4. si no mejora el loss durante `patience` épocas, hace early stopping

## Paso 8: checkpoint guardado
El checkpoint contiene:
- `task`
- `model_name`
- `model_state_dict`
- `config`
- `metadata`

El `metadata` es crucial porque incluye los mapeos necesarios para inferencia:
- `word2idx`
- `label2idx` / `idx2label`
- `tag2idx` / `idx2tag`
- `num_outputs`

---

# 3.2 Evaluación (`evaluate.py`)

`evaluate.py` **no reconstruye todo desde cero de forma arbitraria**. Hace esto:

1. Carga un checkpoint con `load_checkpoint(...)`
2. Lee `task`, `model_name`, `config` y `metadata` desde el checkpoint
3. Reconstruye exactamente la misma arquitectura con `build_model(...)`
4. Carga los pesos con `model.load_state_dict(...)`
5. Construye el dataloader de evaluación con el vocabulario y mappings del checkpoint
6. Ejecuta `test_step(...)`
7. Imprime loss y métrica

## Métricas
- Sentiment: `accuracy`
- NER: `token_accuracy`

### Ojo
En NER se mide **accuracy por token**, no F1 por entidad. Eso hace la evaluación más simple, pero menos representativa que una métrica entity-level.

---

# 3.3 Generación de alertas (`alert_generation.py`)

Este script usa un modelo generativo de Hugging Face:

```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
```

## Cómo funciona

### Paso 1: recibe entradas estructuradas
Por registro necesita:
- `home_team`
- `away_team`
- `sentiment`
- `entities`
- `text`

### Paso 2: transforma la salida de sentimiento
```python
1  -> HOME_WIN
0  -> DRAW
-1 -> AWAY_WIN
```

### Paso 3: agrupa entidades
`group_entities(...)` reorganiza entidades por tipo:
- TEAM
- STADIUM
- PLAYER
- COACH

### Paso 4: construye un resumen NER
Ejemplo:
```text
TEAM: Arsenal and Barcelona
STADIUM: Emirates Stadium
PLAYER: Bukayo Saka and Lewandowski
COACH: Mikel Arteta
```

### Paso 5: construye prompt few-shot
`build_generation_prompt(...)` compone:
- instrucciones del sistema
- reglas de estilo
- ejemplos few-shot
- salida actual de sentimiento
- resumen de NER
- texto del partido o contexto alternativo

### Paso 6: aplica chat template
```python
chat_prompt = self.tokenizer.apply_chat_template(...)
```
Esto convierte los mensajes `system` + `user` al formato esperado por Qwen.

### Paso 7: tokeniza
- truncation=True
- max_length=2048
- padding=True

### Paso 8: genera
La generación usa:
- `max_new_tokens=72`
- `do_sample=True`
- `temperature=0.7`
- `top_p=0.9`
- `no_repeat_ngram_size=3`

### Paso 9: recorta el prompt
Se toma solo la parte generada después del prompt original.

## Punto importante de ejecución
En `NeuralAlertGenerator.__init__()` el modelo Hugging Face se carga así:

```python
self.model = AutoModelForCausalLM.from_pretrained(model_name)
```

No hay un `.to("cuda")` explícito ni `device_map="auto"`.

### Consecuencia
- El generador **normalmente se queda en CPU**.
- Aunque `main.py` mande sentimiento y NER a GPU, **Qwen puede seguir ejecutándose en CPU**.
- Por tanto, el pipeline es “mixto”: SA/NER pueden ir en GPU, pero generación no necesariamente.

Esto es una observación operativa importante para rendimiento.

---

# 3.4 OCR de marcador (`image_captioning.py`)

Aunque el nombre del archivo sugiere “captioning”, realmente **no genera captions**. Lo que hace es OCR para leer el marcador.

## Flujo

### Paso 1: carga imagen
- Usa `PIL` para abrir
- Convierte a RGB
- Luego a BGR para OpenCV

### Paso 2: inicializa EasyOCR
```python
easyocr.Reader(["en"], gpu=use_gpu, verbose=...)
```

### Paso 3: define regiones candidatas
Busca el marcador en varias zonas típicas:
- imagen completa
- centro inferior
- centro medio
- banda inferior
- franja central

### Paso 4: crea variantes preprocesadas
Por cada región genera varias versiones:
- grayscale
- upscale x2
- Otsu threshold
- invertida
- adaptive threshold
- CLAHE

Esto multiplica las oportunidades de OCR correcto.

### Paso 5: OCR sobre cada variante
`run_easyocr(...)` devuelve textos detectados, cajas y confidencias.

### Paso 6: extracción de candidatos de marcador
`extract_score_candidates(...)` intenta encontrar patrones tipo:
- `2-1`
- `2 - 1`
- `2 1`

También combina hits separados como `2`, `-`, `1` o incluso `2` y `1` si están en la misma línea.

### Paso 7: ranking heurístico
`choose_best_candidate(...)` prefiere:
- mayor confianza OCR
- texto más centrado
- cajas más grandes
- regiones probables de scoreboard

### Paso 8: salida final
Normaliza a formato exacto:
```text
2 - 1
```

---

# 3.5 Pipeline completo (`main.py`)

Este es el archivo más importante para entender cómo se conectan todos los módulos.

## Paso 1: parsea argumentos
Parámetros clave:
- `--data-path`
- `--sentiment-checkpoint`
- `--ner-checkpoint`
- `--generator-model-name`
- `--output-path`
- `--mode`
- `--device`
- `--limit`
- `--max-new-tokens`

## Paso 2: resuelve dispositivo
`resolve_device(...)` decide entre:
- CPU
- CUDA
- auto

Ese dispositivo solo aplica a los modelos de sentimiento y NER.

## Paso 3: carga registros
Lee el JSON y opcionalmente limita el número de ejemplos.

## Paso 4: construye pipelines de inferencia

### `SentimentInferencePipeline`
- carga checkpoint
- valida que `task == sentiment`
- reconstruye arquitectura
- carga pesos
- deja el modelo en `.eval()`

### `NERInferencePipeline`
- carga checkpoint
- valida que `task == ner`
- reconstruye arquitectura
- carga pesos
- deja el modelo en `.eval()`

### `NeuralAlertGenerator`
- carga tokenizer Hugging Face
- carga Qwen

## Paso 5: recorre cada registro
Por cada ejemplo del dataset:

### 5.1 valida campos mínimos
Debe existir:
- `text`
- `home_team`
- `away_team`

### 5.2 intenta añadir marcador desde imagen
Si existe `match_id`, busca en `data/`:
- `{match_id}.png`
- `{match_id}.jpg`
- `{match_id}.jpeg`

Si encuentra la imagen:
- lee marcador con `read_score(...)`
- concatena al texto:

```text
The scoreboard image shows a score of X - Y.
```

### Observación importante
Esta línea modifica el registro en memoria:
```python
record["text"] += f" The scoreboard image shows a score of {score}."
```
Por tanto, sentimiento y NER ya trabajan con el texto enriquecido por OCR.

### 5.3 predicción de sentimiento
`sentiment_pipeline.predict(record["text"])`

Hace:
- tokenización
- lookup en vocabulario
- tensorización
- forward
- softmax
- argmax

Devuelve:
- valor predicho (`-1`, `0`, `1`)
- confianza
- distribución completa de probabilidades

### 5.4 predicción de NER
`ner_pipeline.predict(record["text"])`

Hace:
- tokenización con offsets
- conversión a ids
- forward por token
- argmax por token
- decodificación BIO a spans de entidad con `decode_bio_predictions(...)`

Devuelve:
- `tokens`
- `tags`
- `entities`

### 5.5 resolve_sentiment / resolve_entities
Aquí se decide si usar:
- salidas predichas
- o salidas gold del dataset

Depende del `--mode`.

### 5.6 construcción del contexto para generación
`build_generation_context(...)` decide qué texto se pasa al LLM.

#### `predicted`
Pasa `record["text"]`.

#### `gold_sentiment`
Pasa `record["text"]`, pero el sentimiento usado será el gold.

#### `gold_ner`
Pasa `record["text"]`, pero las entidades usadas serán las gold.

#### `structured_only`
Pasa `None`.
Entonces `alert_generation.py` construye contexto solo con datos estructurados.

#### `sentiment_guided`
Construye un texto especial con:
- una pista explícita del resultado esperado
- resumen de entidades
- texto original

Eso fuerza más consistencia entre el resultado predicho y la alerta final.

### 5.7 generación de alerta
`generator.generate_alert(...)`

Entradas finales:
- `home_team`
- `away_team`
- `sentiment` usado
- `entities` usadas
- `original_text` o contexto alternativo

### 5.8 ensamblado de salida
Se guardan:
- entrada original
- `generated_alert`
- `pipeline_mode`
- bloque `pipeline_outputs`

Ese bloque contiene tanto lo predicho como lo realmente usado.

---

# 4. Modelos exactos que usa el sistema

## 4.1 Modelos entrenables del repositorio

### Sentiment
Opciones:
- `meanpool`
- `cnn`
- `bilstm`

#### `MeanPoolClassifier`
- embedding por token
- enmascara padding
- promedio de embeddings válidos
- dropout
- capa lineal final

#### `CNNTextClassifier`
- embedding
- conv1d temporal
- ReLU
- max pooling global
- dropout
- capa lineal final

#### `BiLSTMClassifier`
- embedding
- BiLSTM
- concatena estado final forward + backward
- dropout
- capa lineal final

### NER
Solo:
- `BiLSTMNER`

#### `BiLSTMNER`
- embedding
- BiLSTM bidireccional
- salida por token
- capa lineal a espacio de tags BIO

No hay CRF. La decisión por token es simplemente `argmax` sobre logits.

## 4.2 Modelo generativo
- `Qwen/Qwen2.5-1.5B-Instruct`
- cargado desde Hugging Face con `transformers`
- usado para transformar salidas estructuradas en una frase natural

## 4.3 OCR
- `EasyOCR`
- no es un modelo entrenado en este repo
- se descarga/usa externamente

---

# 5. Cómo se deben ejecutar las cosas, en orden correcto

## Escenario A: quiero entrenar todo desde cero

### 1) crear entorno
```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) instalar dependencias
```bash
pip install torch torchvision torchaudio transformers sentencepiece easyocr opencv-python pillow numpy tqdm tensorboard
```

Si tu instalación de `torch` necesita una variante específica para CUDA, instálala primero siguiendo PyTorch y después ejecuta el resto.

### 3) entrenar sentimiento
```bash
python train.py \
  --task sentiment \
  --data_path train.json \
  --model_name bilstm \
  --epochs 15 \
  --batch_size 8 \
  --save_path models/sentiment_bilstm.pt
```

### 4) entrenar NER
```bash
python train.py \
  --task ner \
  --data_path train.json \
  --model_name bilstm \
  --epochs 15 \
  --batch_size 8 \
  --save_path models/ner_bilstm.pt
```

### 5) evaluar sentimiento
```bash
python evaluate.py \
  --checkpoint models/sentiment_bilstm.pt \
  --data_path train.json \
  --batch_size 8
```

### 6) evaluar NER
```bash
python evaluate.py \
  --checkpoint models/ner_bilstm.pt \
  --data_path train.json \
  --batch_size 8
```

### 7) ejecutar pipeline completo
```bash
python main.py \
  --data-path new_sample.json \
  --sentiment-checkpoint models/sentiment_bilstm.pt \
  --ner-checkpoint models/ner_bilstm.pt \
  --mode predicted \
  --output-path outputs/pipeline_outputs.json
```

## Escenario B: ya tengo checkpoints y solo quiero inferencia

Solo necesitas:
- dataset JSON de entrada
- checkpoint de sentimiento
- checkpoint de NER

Y ejecutar:

```bash
python main.py \
  --data-path new_sample.json \
  --sentiment-checkpoint models/sentiment_bilstm.pt \
  --ner-checkpoint models/ner_bilstm.pt \
  --mode predicted \
  --output-path outputs/pipeline_outputs.json
```

## Escenario C: solo quiero generar alertas desde anotaciones existentes

Si tu JSON ya trae:
- `sentiment`
- `entities`
- `text`
- `home_team`
- `away_team`

puedes usar directamente:

```bash
python alert_generation.py \
  --data-path annotated_matches.json \
  --output-path outputs/generated_alerts.json
```

Esto **no ejecuta sentimiento ni NER entrenados en este repo**. Solo usa el generador LLM.

## Escenario D: solo quiero leer marcadores de imágenes

```bash
python image_captioning.py
```

Con logs OCR detallados:

```bash
python image_captioning.py --verbose
```

---

# 6. Formatos de datos esperados

## 6.1 Dataset para sentimiento
Campos mínimos:

```json
{
  "text": "Manchester City dominated the ball and won comfortably.",
  "sentiment": 1
}
```

También vale:

```json
{
  "text": "Manchester City dominated the ball and won comfortably.",
  "label": 1
}
```

## 6.2 Dataset para NER
Campos mínimos:

```json
{
  "text": "Manchester City beat Chelsea at Etihad Stadium.",
  "entities": [
    { "text": "Manchester City", "label": "TEAM" },
    { "text": "Chelsea", "label": "TEAM" },
    { "text": "Etihad Stadium", "label": "STADIUM" }
  ]
}
```

## 6.3 Dataset para `main.py`
Campos mínimos reales:

```json
[
  {
    "match_id": 1001,
    "home_team": "Arsenal",
    "away_team": "Barcelona",
    "text": "Arsenal controlled long stretches, but Barcelona punished transitions late on."
  }
]
```

## 6.4 Dataset para `alert_generation.py`
Campos mínimos:

```json
[
  {
    "home_team": "Arsenal",
    "away_team": "Barcelona",
    "sentiment": -1,
    "entities": [
      { "text": "Arsenal", "label": "TEAM" },
      { "text": "Barcelona", "label": "TEAM" }
    ],
    "text": "Arsenal controlled possession, but Barcelona were sharper in decisive moments."
  }
]
```

---

# 7. Modos de ejecución de `main.py`

## `predicted`
Usa:
- sentimiento predicho
- entidades predichas

## `gold_sentiment`
Usa:
- sentimiento gold del dataset
- entidades predichas

Requiere campo `sentiment`.

## `gold_ner`
Usa:
- sentimiento predicho
- entidades gold del dataset

Requiere campo `entities`.

## `structured_only`
No pasa el texto original al generador. Solo usa:
- sentimiento
- entidades
- teams
- facts estructurados

## `sentiment_guided`
Inyecta una pista explícita del resultado esperado en el prompt.
Sirve para reducir incoherencias entre clasificación y alerta generada.

---

# 8. Detalles técnicos importantes que conviene saber

## 8.1 El generador no está entrenado en este repo
El modelo Qwen se descarga ya preentrenado desde Hugging Face. Aquí solo se usa para inferencia con prompting.

## 8.2 Sentimiento y NER usan vocabularios propios
Cada checkpoint guarda su propio `word2idx` y mappings. Por eso, al inferir, el texto debe codificarse usando la metadata del checkpoint correcto.

## 8.3 NER usa BIO fijo
Las etiquetas soportadas están fijadas en `utils.py` dentro de `NER_TAGS`.

## 8.4 No hay CRF en NER
El modelo hace predicción por token con `argmax`. Eso simplifica el sistema, pero puede empeorar consistencia estructural frente a una capa CRF.

## 8.5 El OCR modifica el texto de entrada
Si `main.py` encuentra una imagen del partido, añade una frase con el marcador al texto antes de inferir. Eso afecta tanto a sentimiento como a NER y, por tanto, también a la generación final.

## 8.6 El nombre `image_captioning.py` es engañoso
No hace image captioning. Hace OCR de scoreboards.

## 8.7 En NER, `--model_name` no cambia realmente la arquitectura
Aunque desde CLI pongas `--model_name bilstm`, para NER el constructor fuerza `bilstm_ner`.

## 8.8 El generador probablemente corre en CPU
Tal como está escrito, `alert_generation.py` no mueve explícitamente Qwen a CUDA.

---

# 9. Comandos mínimos recomendados

## Instalar dependencias
```bash
pip install torch torchvision torchaudio transformers sentencepiece easyocr opencv-python pillow numpy tqdm tensorboard
```

## Entrenar sentimiento
```bash
python train.py --task sentiment --data_path train.json --model_name bilstm --epochs 15 --batch_size 8 --save_path models/sentiment_bilstm.pt
```

## Entrenar NER
```bash
python train.py --task ner --data_path train.json --model_name bilstm --epochs 15 --batch_size 8 --save_path models/ner_bilstm.pt
```

## Evaluar sentimiento
```bash
python evaluate.py --checkpoint models/sentiment_bilstm.pt --data_path train.json --batch_size 8
```

## Evaluar NER
```bash
python evaluate.py --checkpoint models/ner_bilstm.pt --data_path train.json --batch_size 8
```

## Ejecutar pipeline completo
```bash
python main.py --data-path new_sample.json --sentiment-checkpoint models/sentiment_bilstm.pt --ner-checkpoint models/ner_bilstm.pt --mode predicted --output-path outputs/pipeline_outputs.json
```

## Forzar CPU en inferencia de SA/NER
```bash
python main.py --data-path new_sample.json --sentiment-checkpoint models/sentiment_bilstm.pt --ner-checkpoint models/ner_bilstm.pt --mode predicted --device cpu --output-path outputs/pipeline_outputs.json
```

## Generar alertas desde anotaciones ya preparadas
```bash
python alert_generation.py --data-path annotated_matches.json --output-path outputs/generated_alerts.json
```

## Leer scores desde imágenes
```bash
python image_captioning.py
```

---

# 10. Problemas frecuentes

## Error: faltan dependencias
Instala:
```bash
pip install torch torchvision torchaudio transformers sentencepiece easyocr opencv-python pillow numpy tqdm tensorboard
```

## Error al descargar modelos
Es normal la primera vez para:
- Hugging Face Qwen
- EasyOCR

## `gold_sentiment` falla
Tu dataset no contiene el campo `sentiment`.

## `gold_ner` falla
Tu dataset no contiene el campo `entities` o no es una lista.

## `main.py` no encuentra imagen
Asegúrate de que exista en `data/` un archivo con nombre:
- `{match_id}.png`
- `{match_id}.jpg`
- `{match_id}.jpeg`

## La generación es lenta
Probablemente porque Qwen está ejecutándose en CPU.

## El OCR lee mal el marcador
Es posible si:
- la imagen está borrosa
- el marcador no está en una región visible
- la tipografía es muy pequeña
- el contraste es bajo

---

# 11. Resumen ejecutivo

Este proyecto funciona así:

1. **Entrenas dos modelos propios**:
   - uno de sentimiento
   - uno de NER

2. **Opcionalmente extraes el marcador desde imagen** con EasyOCR.

3. **`main.py` junta todo**:
   - texto enriquecido
   - sentimiento
   - entidades
   - prompt few-shot
   - LLM generativo

4. **La salida final** es un JSON con:
   - predicciones intermedias
   - entidades
   - probabilidades
   - tags BIO
   - alerta final en lenguaje natural

