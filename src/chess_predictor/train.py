"""Entrena el modelo, lo evalúa y lo guarda."""
import json
import logging
import os

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chess_predictor.load_data import clean, load_raw, validate
from chess_predictor.features import build_features

logger = logging.getLogger(__name__)


def train(data_path, n, model_path, metrics_path):
    # 1. Datos
    df = load_raw(data_path)
    validate(df)
    df = clean(df, n)
    X = build_features(df, n)
    y = df["target"]

    #2. Separar entrenamiento (80 %) y test (20 %)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Modelo de referencia: predice siempre la clase más frecuente
    baseline = DummyClassifier().fit(X_train, y_train)

    #esto lo he hecho porque el modelo tonto predice la clase mas frecuentye. en el csv gana el 52% aprox las blancas. 
    # si predice 52% y el modelo predice 53%, no ha aprendido nada útil

    #4. El modelo
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    # 5.Evaluar
    pred = model.predict(X_test)
    metricas = {
        "n": n,
        "partidas_train": len(X_train),
        "partidas_test": len(X_test),
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "accuracy_baseline": round(baseline.score(X_test, y_test), 3),
    }
    logger.info("Métricas: %s", metricas)

    # 6. Guardar modelo y métricas
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    joblib.dump({"model": model, "n": n, "columnas": list(X.columns)}, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metricas, f, indent=2)
    logger.info("Modelo guardado en %s", model_path)

    return metricas