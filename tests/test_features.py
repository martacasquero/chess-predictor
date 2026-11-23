import pandas as pd

from chess_predictor.features import build_features, extraer_variables

PARTIDA = "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O Qf6"


def test_extraer_variables_partida_conocida():
    v = extraer_variables(PARTIDA, n=10)

    assert v["capturas_blancas"] == 1   # Bxc6
    assert v["capturas_negras"] == 1    # dxc6
    assert v["enroque_blancas"] == 1    # O-O
    assert v["enroque_negras"] == 0
    assert v["dama_negras"] == 1        # Qf6
    assert v["peones_blancas"] == 1     # e4
    assert v["peones_negras"] == 3      # e5, a6, dxc6
    assert v["peones_diff"] == -2       # 1 - 3


def test_solo_usa_las_n_primeras_jugadas():
    # Con n=6 el modelo no debe "ver" las capturas de las jugadas 7 y 8
    v = extraer_variables(PARTIDA, n=6)

    assert v["capturas_blancas"] == 0
    assert v["capturas_negras"] == 0
    assert v["enroque_blancas"] == 0


def test_build_features_forma():
    df = pd.DataFrame({"moves": [PARTIDA, "d4 d5 c4 e6"]})
    X = build_features(df, n=4)

    assert X.shape == (2, 15)            # 2 partidas, 15 variables
    assert "capturas_diff" in X.columns