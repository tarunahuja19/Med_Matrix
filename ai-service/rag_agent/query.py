import datetime
import numpy as np
import google.generativeai as genai
from rag_agent.config import get_gemini_api_key
from rag_agent.store import get_disease_chunks
from rag_agent.cache import get_cached_report, set_cached_report

# Hardcoded system prompt template as per architecture specification
SYSTEM_PROMPT = """You are a radiology reporting assistant. Always output reports in exactly this format, no deviations:

---
RADIOLOGY REPORT

Patient: {name} | Age: {age} | Sex: {sex}
Date: {date}

CLINICAL INDICATION
[Why the scan was ordered based on symptoms and disease]

TECHNIQUE
[Typical imaging modality and sequences used for this disease]

FINDINGS
[Organ-by-organ observations relevant to this disease]

IMPRESSION
1. [Most likely diagnosis]
2. [Second differential]
3. [Third differential if applicable]

RECOMMENDATION
[Follow-up scan / biopsy / urgent referral as appropriate]
---

Use only information from the provided research context.
Do not hallucinate findings. If context is insufficient, say so."""

def compute_cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Computes the cosine similarity between two numeric vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(np.dot(a, b) / (norm_a * norm_b))

def generate_radiology_report(
    disease_name: str, 
    patient_metadata: dict, 
    llm_model: str = "gemini-3.5-flash"
) -> str:
    """Orchestrates the RAG query pipeline using Upstash Redis and Gemini API."""
    name = patient_metadata.get("name", "Unknown")
    age = patient_metadata.get("age", "Unknown")
    sex = patient_metadata.get("sex", "Unknown")
    symptoms = patient_metadata.get("symptoms", "None reported")
    date_str = patient_metadata.get("date", datetime.date.today().isoformat())

    # 1. Cache Check
    print(f"Checking Upstash cache for {disease_name} (Age: {age}, Sex: {sex})...")
    cached_report = get_cached_report(disease_name, age, sex)
    if cached_report:
        print("🎉 Cache hit! Returning cached report.")
        # Update name and date placeholders dynamically
        updated_report = cached_report.replace("{name}", name).replace("{date}", date_str)
        return updated_report

    print("Cache miss. Retrieving disease chunks from Upstash Redis...")
    # Fetch all stored sections for the specified disease
    chunks = get_disease_chunks(disease_name)
    if not chunks:
        return f"Error: No reference documents found for the disease '{disease_name}' in Upstash Redis."

    # 2. Embed the query disease_name to compare vectors
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("No GEMINI_API_KEYS are configured in .env. Cannot run query pipeline.")

    print(f"Embedding query disease name with Gemini: '{disease_name}'...")
    genai.configure(api_key=api_key)
    
    embed_response = genai.embed_content(
         model="models/gemini-embedding-001",
         content=disease_name
    )
    query_embedding = embed_response["embedding"]

    # 3. Compute in-memory cosine similarity using NumPy
    print(f"Computing vector similarity for {len(chunks)} sections...")
    scored_chunks = []
    for chunk in chunks:
        stored_emb = chunk.get("embedding")
        if stored_emb:
            similarity = compute_cosine_similarity(query_embedding, stored_emb)
            scored_chunks.append((similarity, chunk))
        else:
            scored_chunks.append((0.0, chunk))

    # Sort chunks by similarity score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Take top-5 most similar chunks
    top_scored_chunks = scored_chunks[:5]
    print("Top matches retrieved:")
    for score, chunk in top_scored_chunks:
        print(f" - [{chunk['section']}] Similarity: {score:.4f}")

    # 4. Concatenate retrieved context
    retrieved_context = ""
    for idx, (score, chunk) in enumerate(top_scored_chunks):
        retrieved_context += f"--- Context Chunk {idx+1} (Section: {chunk['section']}, Source: {chunk['source']}) ---\n"
        retrieved_context += f"{chunk['content']}\n\n"

    # 5. Format Prompts with actual name and date so LLM writes them correctly
    formatted_system_prompt = SYSTEM_PROMPT.format(
        name=name,
        age=age,
        sex=sex,
        date=date_str
    )

    user_message = f"""Disease: {disease_name}
Patient Age: {age}
Patient Sex: {sex}
Symptoms: {symptoms}

Research Reference Context:
{retrieved_context}"""

    # 6. Call Gemini LLM
    print(f"Calling Gemini model {llm_model}...")
    model = genai.GenerativeModel(
        model_name=llm_model,
        system_instruction=formatted_system_prompt
    )
    
    response = model.generate_content(
        user_message,
        generation_config={"temperature": 0.1}
    )
    
    report_output = response.text

    # 7. Store template in cache (retaining literal placeholders)
    print("Caching generated report template in Redis...")
    cache_template = report_output.replace(name, "{name}").replace(date_str, "{date}")
    set_cached_report(disease_name, age, sex, cache_template)

    return report_output
