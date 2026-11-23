# Chess Predictor ♟️

Predicción de **quién ganará una partida de ajedrez (blancas o negras)** a partir de sus **N primeras jugadas**, con una explicación en lenguaje natural generada por IA.

```text
datos (Lichess) → modelo sklearn → pipeline con inferencia → explicación con IA generativa
```

## Inicio rápido

```bash
uv sync                      # instalar dependencias
cp .env.example .env         # configuración (por defecto, IA en modo mock)
uv run pytest                # ejecutar los tests
uv run chess-predictor predict --explain --moves "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Nxd5 Nxf7 Kxf7 Qf3+ Ke6 Nc3 Nb4 a3 Nxc2+ Kd1 Nxa1"
```

Resultado:

```text
Ganador previsto: negras (prob. blancas = 0.29)
[MOCK] Aquí aparecería la explicación de la IA.
```

## Problema

- **Pregunta:** viendo solo las primeras N jugadas de una partida, ¿quién ganará: blancas o negras?
- **Tipo de problema:** clasificación binaria.
- **Variable objetivo (`target`):** se obtiene de la columna `winner`:
  - `1` → ganan las blancas (`white`)
  - `0` → ganan las negras (`black`)
- **Tablas (`draw`):** se eliminan. Son solo 950 de 20.058 partidas (~5 %) y, al quitarlas, la pregunta queda exactamente como "¿quién gana?".

En el dataset hay 10.001 victorias de blancas y 9.107 de negras.

## Datos

- **Archivo:** `data/raw/games.csv`
- **Tamaño:** 20.058 partidas, 16 columnas.
- **Origen:** partidas recogidas de una selección de usuarios de [Lichess.org](https://lichess.org).
- **Licencia:** _pendiente de indicar_.
- **Motivación:** comprobar cuánta información sobre el resultado contienen las primeras jugadas de una partida y en qué momento se "decide".

### Características comprobadas

- Hay **945 partidas duplicadas** (mismo `id`) → se eliminan en la limpieza.
- No hay valores vacíos en `id`, `moves`, `winner` ni `turns`.
- `turns` cuenta **medias jugadas** (coincide con el número de movimientos de `moves`). Por tanto, **N = 20 son 10 jugadas de blancas y 10 de negras**.

## Selección de columnas

El CSV se carga completo, pero el pipeline solo **valida** las columnas que necesita:
si un archivo nuevo de partidas no trae alguna columna que no usamos, el pipeline no debe fallar por ello.

### Columnas que se usan

| Columna | Uso |
|---|---|
| `moves` | De aquí se extraen las N primeras jugadas → **variables del modelo**. |
| `winner` | De aquí sale la **variable objetivo** (1 = ganan blancas, 0 = ganan negras). Las filas con `draw` se eliminan. |
| `turns` | **Solo como filtro**: se usan únicamente las partidas que siguen vivas en la jugada N (`turns > N`). Si la partida ya ha terminado antes, el resultado ya se conoce. |
| `id` | Eliminar duplicados e identificar cada partida. |

### Columnas que no se usan y por qué

| Columnas | Motivo |
|---|---|
| `victory_status`, `last_move_at` | **Data leakage:** solo se conocen cuando la partida ha terminado. |
| `turns` (como variable) | **Data leakage:** indica cuánto duró la partida. Solo se usa como filtro. |
| `white_rating`, `black_rating`, `rated`, `increment_code` | No son trampa, pero la pregunta es **solo con las jugadas**. Con el rating, el modelo se apoyaría sobre todo en "quién es mejor jugador" y no se podría medir cuánto aportan las jugadas. |
| `white_id`, `black_id` | El modelo memorizaría jugadores concretos en lugar de aprender de las jugadas, y no serviría con jugadores nuevos. |
| `opening_eco`, `opening_name`, `opening_ply` | Se derivan de las primeras jugadas: esa información ya está en `moves`. |
| `created_at` | La fecha no aporta información sobre el resultado. |

## Pipeline

```text
games.csv → carga → validación → limpieza → variables → entrenamiento → evaluación → modelo .joblib
            └─────────── data.py ──────────┘  features.py  └──────────── train.py ─────────────┘

partida nueva → modelo .joblib → predicción → explicación con IA
                └────── predict.py ──────┘    └─── llm.py ───┘
```

Todo se ejecuta con un único comando (`cli.py`) y deja registro en `logs/pipeline.log`.

### 1. Carga y validación (`data.py`)

- `load_raw(path)`: lee el CSV; si no existe, lanza `FileNotFoundError` con un mensaje claro.
- `validate(df)`: comprueba que existen las columnas necesarias y que `winner` solo contiene `white`, `black` o `draw`. Si no, lanza `ValueError`.

### 2. Limpieza (`data.py`)

`clean(df, n)`:

1. Elimina partidas duplicadas por `id`.
2. Elimina las tablas.
3. Se queda con las partidas que siguen vivas en la jugada N (`turns > n`).
4. Crea la columna `target` (1 = ganan blancas, 0 = ganan negras).

Partidas que quedan según N:

| N | Partidas |
|---|---|
| 10 | 17.628 |
| 20 | 16.619 |
| 30 | 14.985 |

### 3. Variables (`features.py`)

Se toman solo las N primeras jugadas (`moves.split()[:n]`), se separan las de blancas (posiciones pares) y negras (posiciones impares) y se cuentan, para cada bando:

| Variable | Cómo se detecta | Ejemplo |
|---|---|---|
| `capturas` | la jugada contiene `x` | `Bxc6` |
| `jaques` | la jugada contiene `+` | `Qh5+` |
| `enroque` | la jugada empieza por `O-O` (0 o 1) | `O-O`, `O-O-O` |
| `dama` | la jugada empieza por `Q` | `Qf6` |
| `peones` | la jugada empieza por minúscula | `e4`, `dxc6` |

Para cada una se generan tres columnas: `_blancas`, `_negras` y `_diff` (blancas − negras). En total, **15 variables**.

**Limitación conocida:** todas las capturas cuentan igual (capturar una dama = capturar un peón).

### 4. Entrenamiento y evaluación (`train.py`)

`train(data_path, n, model_path, metrics_path)`:

1. Carga, valida, limpia y genera las variables.
2. Separa **80 % entrenamiento / 20 % test** (`stratify=y`, `random_state=42` → resultado reproducible).
3. Entrena un **modelo de referencia** (`DummyClassifier`): predice siempre la clase más frecuente (ganan blancas).
4. Entrena el modelo: **`StandardScaler` + `LogisticRegression`** dentro de un mismo pipeline de sklearn.
5. Calcula accuracy, F1 y accuracy del modelo de referencia.
6. Guarda:
   - el modelo en `models/modelo_n{N}.joblib` (junto con `n` y los nombres de las columnas, para predecir con el mismo N);
   - las métricas en `reports/metricas_n{N}.json`.

**¿Por qué regresión logística?** Es un problema de clasificación binaria, el modelo es rápido e interpretable (sus coeficientes indican qué variables favorecen a blancas o a negras) y, en una comparación con un Random Forest, obtuvo resultados iguales o ligeramente mejores con estas variables.

### 5. Inferencia (`predict.py`)

- `load_model(path)`: carga el `.joblib`; si no existe, avisa de que hay que entrenar antes.
- `predict(bundle, lista_moves)`: calcula las variables con **la misma función que en el entrenamiento** y devuelve, para cada partida, la probabilidad de que ganen las blancas y el ganador previsto (blancas si la probabilidad es ≥ 0,5).
- El N **sale del propio modelo guardado**, así que no se puede usar por error un N distinto al del entrenamiento.
- Si una partida tiene menos de N jugadas, lanza un `ValueError` claro. Si tiene más, solo se usan las N primeras.

### 6. Explicación con IA generativa (`llm.py`)

La IA traduce la predicción a lenguaje natural, como un comentarista: recibe las N primeras jugadas y la probabilidad, y explica en 3 frases por qué el modelo podría pensar eso.

Todos los clientes tienen la **misma función `generate(prompt)`**, así que son intercambiables y se pueden añadir proveedores nuevos sin tocar el resto del código:

| Proveedor | `LLM_PROVIDER` | Requisitos |
|---|---|---|
| **Mock** (por defecto) | `mock` | Ninguno. Devuelve un texto fijo; se usa para desarrollar y en los tests. |
| **Ollama** | `ollama` | [Ollama](https://ollama.com) instalado y en marcha, con el modelo descargado (`ollama pull llama3.2:3b`). Gratuito y en local. |
| **Hugging Face** | `huggingface` | `HF_TOKEN` (fine-grained, permiso *Make calls to Inference Providers*) **con créditos**. |

**Robustez:**

- Sin `LLM_PROVIDER`, se usa el mock: por defecto nunca se llama a un servicio externo.
- Si el proveedor es desconocido o falta `HF_TOKEN`, se registra un aviso y se usa el mock.
- Si la llamada a la IA falla (Ollama apagado, sin créditos…), se registra el error y se devuelve *"No se ha podido generar la explicación."*: **la predicción se muestra igualmente**.

**Nota sobre Hugging Face:** el cliente está implementado y la llamada llega a la API, pero devuelve `402 Payment Required` porque las cuentas gratuitas ya no incluyen créditos para Inference Providers. Por eso el proveedor real utilizado es **Ollama**.

## Resultados

Con N = 20:

```json
{
  "n": 20,
  "partidas_train": 13295,
  "partidas_test": 3324,
  "accuracy": 0.561,
  "f1": 0.626,
  "accuracy_baseline": 0.519
}
```

Evolución según N:

| N (medias jugadas) | Accuracy modelo | Accuracy baseline |
|---|---|---|
| 10 | 0,538 | 0,522 |
| 20 | 0,561 | 0,519 |
| 30 | 0,616 | 0,517 |
| 40 | **0,677** | 0,513 |

**Conclusión:** cuantas más jugadas ve el modelo, mejor predice. Con pocas jugadas la partida apenas está decidida.

**Sobre N:** subir N mejora la accuracy, pero hace la pregunta más fácil (la partida ya está más decidida) y deja menos partidas. Por eso N se mantiene como parámetro (por defecto, 20) y el resultado principal es la **curva de accuracy según N**. Para mejorar la accuracy sin subir N, la vía es mejorar las variables (ver [Próximos pasos](#próximos-pasos-proyecto-iii)).

**Sobre las métricas:**

- La métrica de referencia es la **accuracy comparada con el baseline**: el modelo solo aporta valor en la medida en que supera al modelo que predice siempre "ganan blancas".
- El **F1 no es fiable aquí** para comparar con el baseline: el `DummyClassifier` saca un F1 de ~0,68 porque predice siempre la clase 1, lo que infla esa métrica.

## Instalación

El proyecto usa [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

También hay un `requirements.txt` (generado con `uv export`) para instalar con pip:

```bash
pip install -r requirements.txt
```

## Configuración

Copia la plantilla y edítala si quieres usar una IA real:

```bash
cp .env.example .env
```

| Variable | Por defecto | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock`, `ollama` o `huggingface` |
| `OLLAMA_URL` | `http://localhost:11434` | Dirección del servidor de Ollama |
| `OLLAMA_MODEL` | `llama3.2:3b` | Modelo de Ollama |
| `HF_TOKEN` | _(vacío)_ | Token de Hugging Face |
| `HF_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Modelo de Hugging Face |

El archivo `.env` **no se sube al repositorio** (está en `.gitignore`).

## Uso

```bash
uv run chess-predictor --help
```

### Entrenar

```bash
uv run chess-predictor train --n 20
```

Opciones: `--n` (medias jugadas, por defecto 20) y `--data` (ruta del CSV, por defecto `data/raw/games.csv`).

El repositorio incluye ya entrenado el modelo de N = 20 (`models/modelo_n20.joblib`) y sus métricas, así que se puede predecir sin entrenar. Hay que volver a entrenar cuando:

- se añaden partidas nuevas al CSV;
- cambian las variables (`features.py`);
- cambia el modelo o sus parámetros;
- se quiere usar otro N (cada N es un modelo distinto).

### Predecir

```bash
uv run chess-predictor predict --moves "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Nxd5 Nxf7 Kxf7 Qf3+ Ke6 Nc3 Nb4 a3 Nxc2+ Kd1 Nxa1"
```

Con explicación de la IA:

```bash
uv run chess-predictor predict --explain --moves "..."
```

Opciones: `--moves` (jugadas separadas por espacios, obligatorio), `--n` (N del modelo a usar, por defecto 20) y `--explain`.

### Logs y errores

- Todos los pasos se registran en consola y en `logs/pipeline.log`, con fecha y hora.
- Los errores esperables (CSV o modelo inexistente, partida demasiado corta) se muestran como una línea de `ERROR` clara y el comando termina con código de salida `1`.

## Tests

```bash
uv run pytest -v                     # todos (16)
uv run pytest -m "not integration"   # solo unitarios (15)
uv run pytest -m integration         # solo integración (1)
```

| Archivo | Tipo | Qué comprueba |
|---|---|---|
| `test_data.py` | Unitarios | Carga, validación (columnas, `winner`) y limpieza (duplicados, tablas, filtro `turns`, `target`). |
| `test_features.py` | Unitarios | Variables de una partida contada a mano y **test anti-leakage**: solo se usan las N primeras jugadas. |
| `test_llm.py` | Unitarios | Elección del cliente según `LLM_PROVIDER` y que un fallo de la IA no rompe el pipeline. No necesitan internet, Ollama ni token. |
| `test_integracion.py` | Integración | Entrenar → guardar → cargar → predecir, con un CSV pequeño de 40 partidas en una carpeta temporal. |

## Estructura

```text
chess-predictor/
├── data/raw/games.csv
├── models/modelo_n20.joblib    # modelo entrenado (N = 20)
├── reports/metricas_n20.json   # métricas
├── logs/                       # logs del pipeline (no se suben)
├── src/chess_predictor/
│   ├── __init__.py
│   ├── data.py                 # carga, validación y limpieza
│   ├── features.py             # variables a partir de las N primeras jugadas
│   ├── train.py                # entrenamiento, evaluación y guardado
│   ├── predict.py              # carga del modelo e inferencia
│   ├── llm.py                  # clientes de IA generativa (mock, Ollama, Hugging Face)
│   └── cli.py                  # comando chess-predictor (train / predict)
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_llm.py
│   └── test_integracion.py
├── .env.example
├── pyproject.toml
├── requirements.txt
├── AI_USAGE.md
└── README.md
```

## Próximos pasos (Proyecto III)

La estructura es modular: cada paso está en su propio módulo, así que lo siguiente se añade **sin reescribir el núcleo**.

- **API / app:** endpoint de predicción que reutiliza `load_model` y `predict`, y endpoint de explicación que reutiliza `explain` y `get_client`.
- **Mejora del modelo:** añadir el **material** de cada bando con la librería `chess` (solo afecta a `features.py`) y comparar modelo v1 vs. v2 con la curva de accuracy según N.
- **Comparación con rating:** entrenar un modelo con jugadas + rating para medir cuánto aportan las jugadas por sí mismas.
- **CI:** ejecutar los tests automáticamente en cada push.
- **Despliegue y monitorización.**

## Uso de IA en el desarrollo

Ver [AI_USAGE.md](AI_USAGE.md).