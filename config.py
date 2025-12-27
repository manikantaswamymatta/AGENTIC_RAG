# Configuration management for Hugging Face client

import os
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from huggingface_hub import InferenceClient

load_dotenv()


class HuggingFaceConfig(BaseModel):
    """
    Configuration model for Hugging Face Inference API
    """
    api_token: str
    chat_model: str

    @field_validator("api_token", "chat_model")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Configuration value cannot be empty")
        return v


def get_hf_config() -> HuggingFaceConfig:
    """
    Load Hugging Face configuration from environment variables
    """
    try:
        cfg = HuggingFaceConfig(
            api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
            chat_model=os.getenv(
                "HF_CHAT_MODEL",
                "mistralai/Mistral-7B-Instruct-v0.2"
            ),
        )

        print("[CONFIG] Loaded HuggingFaceConfig ->", cfg.model_dump())
        return cfg

    except Exception as exc:
        print("[CONFIG ERROR] Failed to load HuggingFaceConfig:", exc)
        raise


def get_hf_client(cfg: HuggingFaceConfig) -> InferenceClient:
    """
    Create Hugging Face inference client
    """
    print("[CONFIG] Creating Hugging Face InferenceClient...")

    try:
        client = InferenceClient(
            model=cfg.chat_model,
            token=cfg.api_token,
            timeout=120
        )

        print("[CONFIG] Hugging Face client created successfully.")
        return client

    except Exception as exc:
        print("[CONFIG ERROR] Failed to create Hugging Face client:", exc)
        raise
