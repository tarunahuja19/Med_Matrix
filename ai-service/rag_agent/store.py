import json
from rag_agent.cache import redis_client

def make_doc_key(disease_name: str) -> str:
    """Constructs the Redis key for storing reference documents of a disease."""
    disease_norm = disease_name.strip().lower().replace(" ", "_")
    return f"med_docs:{disease_norm}"

def store_disease_chunks(disease_name: str, chunks: list) -> bool:
    """Stores the complete list of section chunks (with content, metadata, and embeddings) in Redis."""
    if not redis_client:
        print("⚠️ Upstash Redis client is not initialized.")
        return False
        
    key = make_doc_key(disease_name)
    try:
        # Serialize list of dicts containing section, content, source, and embedding
        serialized_data = json.dumps(chunks)
        redis_client.set(key, serialized_data)
        return True
    except Exception as e:
        print(f"⚠️ Error storing document chunks for {disease_name} in Redis: {e}")
        return False

def get_disease_chunks(disease_name: str) -> list:
    """Retrieves and deserializes the list of section chunks for a disease from Redis."""
    if not redis_client:
        print("⚠️ Upstash Redis client is not initialized.")
        return []
        
    key = make_doc_key(disease_name)
    try:
        data = redis_client.get(key)
        if data:
            # Parse dynamic response type from Upstash (string or bytes)
            parsed_str = data.decode("utf-8") if isinstance(data, bytes) else str(data)
            return json.loads(parsed_str)
        return []
    except Exception as e:
        print(f"⚠️ Error retrieving document chunks for {disease_name} from Redis: {e}")
        return []

def clear_all_documents() -> bool:
    """Finds and deletes all med_docs:* keys in Upstash Redis."""
    if not redis_client:
        print("⚠️ Upstash Redis client is not initialized.")
        return False
    try:
        # Standard scan of keys matching pattern
        # Since Upstash Redis SCAN returns a list of keys, we can delete them
        cursor = 0
        keys_to_delete = []
        
        # Simple loop to scan keys matching med_docs:*
        # Upstash SCAN returns: [new_cursor, [keys]]
        res = redis_client.scan(cursor=cursor, match="med_docs:*", count=100)
        if res and len(res) == 2:
            keys_to_delete.extend(res[1])
            
        if keys_to_delete:
            print(f"Deleting keys: {keys_to_delete}")
            redis_client.delete(*keys_to_delete)
            print(f"Cleared {len(keys_to_delete)} document keys.")
        else:
            print("No document keys found to clear.")
        return True
    except Exception as e:
        print(f"⚠️ Error clearing document keys: {e}")
        return False
