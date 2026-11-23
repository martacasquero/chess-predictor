#convierte las jugadas que son txto en variables numéricas 
import pandas as pd


def contar(jugadas):
    #Cuenta capturas, jaques, enroque, jugadas de dama y de peón.
    return {
        "capturas": sum("x" in j for j in jugadas),
        "jaques": sum("+" in j for j in jugadas),
        "enroque": int(any(j.startswith("O-O") for j in jugadas)),
        "dama": sum(j.startswith("Q") for j in jugadas),
        "peones": sum(j[0].islower() for j in jugadas),
    }


def extraer_variables(moves, n):
    #Variables de una partida usando solo sus n primeras jugadas
    jugadas = moves.split()[:n]
    blancas = contar(jugadas[0::2])
    negras = contar(jugadas[1::2])

    fila = {}
    for nombre in blancas:
        fila[f"{nombre}_blancas"] = blancas[nombre]
        fila[f"{nombre}_negras"] = negras[nombre]
        fila[f"{nombre}_diff"] = blancas[nombre] - negras[nombre]
    return fila


def build_features(df, n):
    #Aplica extraer_variables a todas las partidas -> tabla X para el modelo, que lo usa para aprender. 
    filas = [extraer_variables(m, n) for m in df["moves"]]
    return pd.DataFrame(filas)

