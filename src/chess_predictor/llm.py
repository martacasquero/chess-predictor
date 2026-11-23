

#Cliente de IA generativa: explica la predicción en lenguaje natural.
import logging
import os

logger = logging.getLogger(__name__)

def build_prompt(moves, n, prob_blancas):

    return (
        "Eres un comentarista de ajedrez. Un modelo ha analizado "
        f"las {n} primeras medias jugadas de esta partida:\n{moves}\n\n"
        f"Probabilidad de que ganen las blancas: {prob_blancas:.0%}\n\n"
        "Explica en 3 frases, en español, por qué el modelo podría pensar eso."
    )

class MockClient:
    #IA de mentira: no se conecta a nada y siempre responde lo mismo

    def generate(self, prompt):
        return "[MOCK] Aquí aparecería la explicación de la IA."


def get_client():
    """Elige el cliente según la variable de entorno LLM_PROVIDER."""
    proveedor = os.getenv("LLM_PROVIDER", "mock")

    if proveedor == "mock":
        logger.info("Usando el cliente mock")
        return MockClient()

    logger.warning("Proveedor '%s' desconocido: se usa el mock", proveedor)
    return MockClient()