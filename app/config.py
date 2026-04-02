from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Deepgram
    deepgram_api_key: str

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"

    # ElevenLabs
    elevenlabs_api_key: str
    elevenlabs_voice_id: str

    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""

    # Agent
    system_prompt: str = (
        "You are a helpful customer service agent. "
        "Be concise and friendly. Keep responses short since this is a phone call."
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
