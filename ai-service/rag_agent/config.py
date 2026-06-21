import os
from dotenv import load_dotenv

# Load .env file from root of workspace if it exists
# Moving up one directory to reach the workspace root where .env resides
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path = os.path.join(root_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()


# Upstash Redis Credentials
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

# API Keys
GEMINI_API_KEYS = []
keys_str = os.getenv("GEMINI_API_KEYS", "")
if keys_str:
    GEMINI_API_KEYS = [k.strip().strip('"\'') for k in keys_str.split(",") if k.strip()]
else:
    fallback_key = os.getenv("GEMINI_API_KEY")
    if fallback_key:
        GEMINI_API_KEYS = [fallback_key.strip().strip('"\'')]

_key_index = 0

def get_gemini_api_key() -> str:
    """Returns the next available Gemini API key from the rotation pool to avoid rate limits."""
    global _key_index
    if not GEMINI_API_KEYS:
        return ""
    key = GEMINI_API_KEYS[_key_index]
    _key_index = (_key_index + 1) % len(GEMINI_API_KEYS)
    return key

# Check if keys are loaded
def validate_config():
    missing = []
    if not UPSTASH_REDIS_REST_URL:
        missing.append("UPSTASH_REDIS_REST_URL")
    if not UPSTASH_REDIS_REST_TOKEN:
        missing.append("UPSTASH_REDIS_REST_TOKEN")
    if not GEMINI_API_KEYS:
        missing.append("GEMINI_API_KEYS")
    
    if missing:
        print(f"⚠️ Warning: Missing environment variables: {', '.join(missing)}")
        return False
    return True
