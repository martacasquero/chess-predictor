import pandas as pd
import pytest

from chess_predictor.load_data import clean, load_raw, validate


@pytest.fixture
def partidas():
    """Un mini-dataset de 4 partidas inventadas para los tests."""
    return pd.DataFrame({
        "id": ["a", "b", "c", "a"],
        "turns": [30, 30, 5, 30],
        "winner": ["white", "black", "white", "white"],
        "moves": ["e4 e5", "d4 d5", "e4 e5", "e4 e5"],
    })


def test_validate_ok(partidas):
    validate(partidas)  # si no lanza error, el test pasa


def test_validate_falta_columna(partidas):
    sin_moves = partidas.drop(columns="moves")
    with pytest.raises(ValueError):
        validate(sin_moves)


def test_validate_winner_raro(partidas):
    partidas.loc[0, "winner"] = "patata"
    with pytest.raises(ValueError):
        validate(partidas)


def test_load_raw_archivo_no_existe():
    with pytest.raises(FileNotFoundError):
        load_raw("no_existe.csv")


def test_clean(partidas):
    partidas.loc[1, "winner"] = "draw"
    resultado = clean(partidas, n=20)

    # 'a' duplicada -> 1 vez, 'b' tablas -> fuera, 'c' 5 jugadas -> fuera
    assert list(resultado["id"]) == ["a"]
    assert list(resultado["target"]) == [1]