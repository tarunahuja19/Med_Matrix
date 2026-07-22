import json
from upstash_redis import Redis
from rag_agent.config import UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN

# Initialize Upstash Redis client
redis_client = None
if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
    # Upstash REST URL sometimes contains quotes if loaded directly, strip them
    url = UPSTASH_REDIS_REST_URL.strip('"\'')
    token = UPSTASH_REDIS_REST_TOKEN.strip('"\'')
    redis_client = Redis(url=url, token=token)

def get_age_bucket(age: int) -> str:
    """Helper to convert patient age into discrete buckets for cache efficiency."""
    try:
        age_int = int(age)
    except (ValueError, TypeError):
        return "unknown"
        
    if age_int < 18:
        return "pediatric"
    elif age_int < 35:
        return "young_adult"
    elif age_int < 55:
        return "middle_aged"
    elif age_int < 75:
        return "senior"
    else:
        return "geriatric"

def make_cache_key(disease_name: str, age: int, sex: str) -> str:
    """Constructs cache key from disease name, age bucket, and sex."""
    age_bucket = get_age_bucket(age)
    # Normalize values for consistency
    disease_norm = disease_name.strip().lower().replace(" ", "_")
    sex_norm = sex.strip().lower()
    return f"rad_report:{disease_norm}:{age_bucket}:{sex_norm}"

def get_cached_report(disease_name: str, age: int, sex: str) -> str:
    """Retrieves cached radiology report if available, else returns None."""
    if not redis_client:
        return None
    
    key = make_cache_key(disease_name, age, sex)
    try:
        report = redis_client.get(key)
        if report:
            # Handle byte decoding or JSON decoding if needed, but it's stored as plain string
            return report.decode("utf-8") if isinstance(report, bytes) else str(report)
    except Exception as e:
        print(f"⚠️ Redis cache read error: {e}")
    return None

def set_cached_report(disease_name: str, age: int, sex: str, report_content: str, ttl_seconds: int = 86400):
    """Caches generated radiology report with an expiration TTL (default 24 hours)."""
    if not redis_client:
        return
    
    key = make_cache_key(disease_name, age, sex)
    try:
        # upstash-redis set supports ex parameter for TTL
        redis_client.set(key, report_content, ex=ttl_seconds)
    except Exception as e:
        print(f"⚠️ Redis cache write error: {e}")
