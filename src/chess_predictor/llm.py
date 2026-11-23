

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
    
    if proveedor == "huggingface":
        token = os.getenv("HF_TOKEN")
        if not token:
            logger.warning("Falta HF_TOKEN: se usa el mock")
            return MockClient()
        modelo = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        logger.info("Usando Hugging Face (%s)", modelo)
        return HuggingFaceClient(token, modelo)
    
    if proveedor == "ollama":
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        modelo = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        logger.info("Usando Ollama (%s)", modelo)
        return OllamaClient(url, modelo)
    
    logger.warning("Proveedor '%s' desconocido: se usa el mock", proveedor)
    return MockClient()

class HuggingFaceClient:
    """IA real: llama a un modelo de lenguaje de Hugging Face."""

    def __init__(self, token, modelo):
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(api_key=token)
        self.modelo = modelo

    def generate(self, prompt):
        respuesta = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.modelo,
            max_tokens=300,
        )
        return respuesta.choices[0].message.content


class OllamaClient:
    """IA real pero la he cambiado porque hugging face era de pago: llama a un modelo que corre en tu Mac con Ollama."""

    def __init__(self, url, modelo):
        self.url = url
        self.modelo = modelo

    def generate(self, prompt):
        import requests

        respuesta = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.modelo,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        respuesta.raise_for_status()
        return respuesta.json()["message"]["content"]