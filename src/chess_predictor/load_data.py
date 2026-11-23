"""Carga y validación de los datos."""
import logging #para logs 
import os #para manejar rutas de archivos 

import pandas as pd #para manejar dataframes 

logger = logging.getLogger(__name__)

COLUMNAS = ["id", "turns", "winner", "moves"]
GANADORES = {"white", "black", "draw"} #en la limpieza voy a quitar draw y convertir white y black en 1/0

def load_raw(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    df = pd.read_csv(path)
    logger.info("Cargadas %d filas", len(df))
    return df


def validate(df):
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}")
    if not set(df["winner"].unique()) <= GANADORES:
        raise ValueError("La columna 'winner' tiene valores raros")
    logger.info("Validación OK")


def clean(df, n):
    #Deja solo las partidas útiles para predecir con las n primeras jugadas
    total = len(df)

    df = df.drop_duplicates(subset="id")
    df = df[df["winner"] != "draw"]
    df = df[df["turns"] > n]

    df = df.copy()
    df["target"] = (df["winner"] == "white").astype(int)

    logger.info("Limpieza: %d -> %d partidas (n=%d)", total, len(df), n)
    return df.reset_index(drop=True)