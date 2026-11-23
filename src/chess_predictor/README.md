# Chess Predictor ♟️

Predicción de **quién ganará una partida de ajedrez (blancas o negras)** a partir de sus **N primeras jugadas**.

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
| `id` | Eliminar duplicados e identificar cada partida al hacer predicciones. |

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
```

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

**Limitación conocida:** todas las capturas cuentan igual (capturar una dama = capturar un peón). Se podrá mejorar añadiendo el **material** de cada bando con la librería `chess`.

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

**Sobre las métricas:**

- La métrica de referencia es la **accuracy comparada con el baseline**: el modelo solo aporta valor en la medida en que supera al modelo que predice siempre "ganan blancas".
- El **F1 no es fiable aquí** para comparar con el baseline: el `DummyClassifier` saca un F1 de ~0,68 porque predice siempre la clase 1, lo que infla esa métrica.

## Instalación

El proyecto usa [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Uso

Entrenar el modelo con N = 20:

```bash
uv run python -c "
import logging; logging.basicConfig(level='INFO')
from chess_predictor.train import train
train('data/raw/games.csv', 20, 'models/modelo_n20.joblib', 'reports/metricas_n20.json')
"
```

Hay que volver a entrenar cuando:

- se añaden partidas nuevas al CSV;
- cambian las variables (`features.py`);
- cambia el modelo o sus parámetros;
- se quiere usar otro N (cada N es un modelo distinto).

## Estructura

```text
chess-predictor/
├── data/raw/games.csv
├── models/                  # modelos entrenados (.joblib)
├── reports/                 # métricas (.json)
├── logs/
├── src/chess_predictor/
│   ├── __init__.py
│   ├── data.py              # carga, validación y limpieza
│   ├── features.py          # variables a partir de las N primeras jugadas
│   └── train.py             # entrenamiento, evaluación y guardado
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Próximos pasos

- [ ] Inferencia: cargar el modelo y predecir partidas nuevas.
- [ ] Comando único para ejecutar el pipeline, con logs a fichero.
- [ ] Cliente de IA generativa (modo `mock` + proveedor real).
- [ ] Tests con pytest.
- [ ] `.env.example` y `AI_USAGE.md`.
- [ ] Mejora: variable de material con la librería `chess`.