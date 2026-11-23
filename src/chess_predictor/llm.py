

#Cliente de IA generativa: explica la predicción en lenguaje natural.

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