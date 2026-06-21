import os
import google.generativeai as genai
import logging

logger = logging.getLogger("rag-agent")

def get_gemini_keys():
    """Reads GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY from environment."""
    keys_str = os.getenv("GEMINI_API_KEYS", "")
    if not keys_str:
        single_key = os.getenv("GEMINI_API_KEY", "")
        if single_key:
            return [single_key]
        return []
    return [k.strip() for k in keys_str.split(",") if k.strip()]

def call_gemini_with_retry(api_func, *args, **kwargs):
    """
    Calls a Gemini API function, rotating through GEMINI_API_KEYS if a rate limit
    or other API error occurs.
    """
    keys = get_gemini_keys()
    if not keys:
        raise ValueError("No Gemini API keys found in environment (GEMINI_API_KEYS or GEMINI_API_KEY)")
    
    last_error = None
    for attempt, key in enumerate(keys):
        try:
            logger.info(f"Configuring Gemini with key index {attempt}...")
            genai.configure(api_key=key)
            return api_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Gemini API call failed with key index {attempt}: {e}. Retrying with next key...")
            last_error = e
            
    raise last_error or RuntimeError("Gemini API call failed after retrying all configured keys")
