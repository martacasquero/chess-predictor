"""Punto de entrada del pipeline: uv run chess-predictor <comando>."""
import argparse
import logging
import os
import sys

from chess_predictor.predict import load_model, predict
from chess_predictor.train import train

logger = logging.getLogger("chess_predictor")


def setup_logging():
    """Logs a la consola y al fichero logs/pipeline.log."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/pipeline.log")],
    )


def main():
    parser = argparse.ArgumentParser(description="Predice quién gana una partida de ajedrez")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_train = sub.add_parser("train", help="Entrena y guarda el modelo")
    p_train.add_argument("--n", type=int, default=20, help="Número de medias jugadas")
    p_train.add_argument("--data", default="data/raw/games.csv")

    p_pred = sub.add_parser("predict", help="Predice el ganador de una partida")
    p_pred.add_argument("--moves", required=True, help="Jugadas separadas por espacios")
    p_pred.add_argument("--n", type=int, default=20, help="N del modelo a usar")

    args = parser.parse_args()
    setup_logging()

    model_path = f"models/modelo_n{args.n}.joblib"
    try:
        if args.comando == "train":
            train(args.data, args.n, model_path, f"reports/metricas_n{args.n}.json")
        elif args.comando == "predict":
            resultado = predict(load_model(model_path), [args.moves])
            fila = resultado.iloc[0]
            print(f"Ganador previsto: {fila['ganador']} (prob. blancas = {fila['prob_blancas']})")
    except (FileNotFoundError, ValueError) as e:
        logger.error(e)
        sys.exit(1)