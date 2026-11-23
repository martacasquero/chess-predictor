#Carga el modelo guardado y predice quién gana
import logging
import os

import joblib
import pandas as pd

from chess_predictor.features import build_features

logger = logging.getLogger(__name__)


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo: {model_path}. ¿Has entrenado antes?")
    bundle = joblib.load(model_path)
    logger.info("Modelo cargado desde %s (n=%d)", model_path, bundle["n"])
    return bundle


def predict(bundle, lista_moves):
    """Recibe una lista de partidas (texto de jugadas) y devuelve una tabla con la predicción."""
    n = bundle["n"]
    for moves in lista_moves:
        if len(moves.split()) < n:
            raise ValueError(f"La partida tiene menos de {n} jugadas: '{moves}'")

    X = build_features(pd.DataFrame({"moves": lista_moves}), n)
    prob_blancas = bundle["model"].predict_proba(X)[:, 1]

    resultado = pd.DataFrame({
        "moves": lista_moves,
        "prob_blancas": prob_blancas.round(3),
        "ganador": ["blancas" if p >= 0.5 else "negras" for p in prob_blancas],
    })
    logger.info("Predichas %d partidas", len(resultado))
    return resultado