from pydantic_ai import ModelProfile

gemma3_12b_profile = ModelProfile(
    supports_tools=False,
    supports_json_schema_output=True,
    default_structured_output_mode='native',
)

MODEL_PROFILES = {
    'huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf:latest': gemma3_12b_profile
}

def get_model_profile(model_name: str) -> ModelProfile:
    """Return the profile for a given model if it exists, otherwise None."""
    return MODEL_PROFILES.get(model_name)
