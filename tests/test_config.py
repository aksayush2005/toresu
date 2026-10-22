import pytest
from toresu import Settings

def test_settings_allow_gemini_to_be_unconfigured(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert Settings.from_env() == Settings(gemini_api_key=None, gemini_model=None)


@pytest.mark.parametrize(
    ("api_key", "model"),
    [("test-key", None), (None, "test-model")],
)
def test_settings_reject_partial_gemini_configuration(monkeypatch, tmp_path, api_key, model):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", api_key or "")
    monkeypatch.setenv("GEMINI_MODEL", model or "")

    with pytest.raises(ValueError, match="must be set together"):
        Settings.from_env()


def test_settings_keep_configured_model_identifier_as_supplied(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "configured-model-id")

    settings = Settings.from_env()

    assert settings.gemini_api_key == "test-key"
    assert settings.gemini_model == "configured-model-id"