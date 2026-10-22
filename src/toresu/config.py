import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        api_key = _optional_value("GEMINI_API_KEY")
        model = _optional_value("GEMINI_MODEL")

        if (api_key is None) != (model is None):
            raise ValueError("GEMINI_API_KEY and GEMINI_MODEL must be set together.")

        return cls(gemini_api_key=api_key, gemini_model=model)


def _optional_value(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None