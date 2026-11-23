from chess_predictor.llm import (
    MockClient,
    OllamaClient,
    build_prompt,
    explain,
    get_client,
)


def test_build_prompt_incluye_los_datos():
    prompt = build_prompt("e4 e5 Nf3", 3, 0.29)
    assert "e4 e5 Nf3" in prompt
    assert "29%" in prompt


def test_mock_responde_siempre_lo_mismo():
    assert MockClient().generate("cualquier cosa").startswith("[MOCK]")


def test_get_client_por_defecto_es_mock(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert isinstance(get_client(), MockClient)


def test_get_client_proveedor_desconocido_usa_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "patata")
    assert isinstance(get_client(), MockClient)


def test_get_client_huggingface_sin_token_usa_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert isinstance(get_client(), MockClient)


def test_get_client_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert isinstance(get_client(), OllamaClient)


class ClienteRoto:
    """Cliente falso que siempre falla, como si Ollama estuviera apagado."""

    def generate(self, prompt):
        raise ConnectionError("Ollama apagado")


def test_explain_no_rompe_si_la_ia_falla():
    texto = explain(ClienteRoto(), "e4 e5", 2, 0.5)
    assert texto == "No se ha podido generar la explicación."