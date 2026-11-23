import json

import pandas as pd
import pytest

from chess_predictor.predict import load_model, predict
from chess_predictor.train import train

JUGADAS = [
    "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6",
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6",
    "e4 e6 d4 d5 Nc3 Bb4 e5 c5",
]


@pytest.fixture
def csv_pequeno(tmp_path):
    """CSV de 40 partidas inventadas, guardado en una carpeta temporal."""
    filas = []
    for i in range(40):
        filas.append({
            "id": f"p{i}",
            "turns": 30,
            "winner": "white" if i % 2 == 0 else "black",
            "moves": JUGADAS[i % 4],
        })
    ruta = tmp_path / "games.csv"
    pd.DataFrame(filas).to_csv(ruta, index=False)
    return ruta


@pytest.mark.integration
def test_entrenar_guardar_cargar_y_predecir(csv_pequeno, tmp_path):
    model_path = str(tmp_path / "models" / "modelo.joblib")
    metrics_path = str(tmp_path / "reports" / "metricas.json")

    # 1. Entrenar y guardar
    train(str(csv_pequeno), 4, model_path, metrics_path)

    # 2. Las métricas se han guardado
    with open(metrics_path) as f:
        metricas = json.load(f)
    assert 0 <= metricas["accuracy"] <= 1

    # 3. Cargar el modelo y predecir
    modelo = load_model(model_path)
    assert modelo["n"] == 4

    resultado = predict(modelo, ["e4 e5 Nf3 Nc6 Bb5"])
    assert resultado.loc[0, "ganador"] in {"blancas", "negras"}
    assert 0 <= resultado.loc[0, "prob_blancas"] <= 1