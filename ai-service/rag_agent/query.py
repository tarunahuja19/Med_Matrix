import os
import json
import logging
import re
from datetime import datetime
import numpy as np
import google.generativeai as genai
from rag_agent.gemini_client import call_gemini_with_retry
from rag_agent.ingest import get_redis_client, run_single_file_ingestion

logger = logging.getLogger("rag-agent-query")

def get_age_bucket(age: int) -> str:
    """Classifies age into demographic buckets."""
    if age <= 12:
        return "child"
    elif age <= 17:
        return "adolescent"
    elif age <= 35:
        return "young_adult"
    elif age <= 65:
        return "adult"
    else:
        return "senior"

def get_patient_age(patient_metadata: dict) -> int:
    """Retrieves or calculates age from patient metadata."""
    age = patient_metadata.get("age")
    if age is not None:
        try:
            return int(age)
        except ValueError:
            pass
            
    # Calculate from dateOfBirth if age is not directly provided
    dob_str = patient_metadata.get("dateOfBirth")
    if dob_str:
        try:
            # Handle ISO timestamp format (e.g. 2026-06-21T00:00:00.000Z)
            dob_clean = dob_str.split("T")[0]
            dob = datetime.strptime(dob_clean, "%Y-%m-%d")
            today = datetime.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except Exception as e:
            logger.warning(f"Failed to parse dateOfBirth '{dob_str}': {e}")
            
    return 40  # Fallback default age

def cosine_similarity(a, b):
    """Computes cosine similarity between two 1D vectors."""
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def find_matching_markdown_file(disease_name: str, data_dir: str = "/app/data") -> str | None:
    """Attempts to find a matching markdown file for self-healing ingestion."""
    # Check current workspace paths
    possible_paths = [
        data_dir,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
        "data"
    ]
    
    clean_query = disease_name.lower().strip().replace(" ", "_")
    
    for path in possible_paths:
        if not os.path.exists(path):
            continue
        for f in os.listdir(path):
            if f.endswith(".md"):
                name_without_ext = os.path.splitext(f)[0].lower().replace(" ", "_")
                # Direct match or query match
                if name_without_ext == clean_query or name_without_ext.replace("_", "") == clean_query.replace("_", ""):
                    return os.path.join(path, f)
                    
    return None

def fetch_document_chunks(disease_name: str, redis_client) -> list:
    """Fetches documents from Redis, falling back to local ingestion if missing."""
    redis_key = f"med_docs:{disease_name}"
    val = redis_client.get(redis_key)
    
    if val:
        logger.info(f"Retrieved document chunks for '{disease_name}' from Redis.")
        return json.loads(val)
        
    # Self-healing: Cache miss on reference data. Try to ingest the markdown file dynamically.
    logger.warning(f"Cache miss for document chunks '{redis_key}'. Running self-healing ingestion...")
    matching_file = find_matching_markdown_file(disease_name)
    if matching_file:
        try:
            run_single_file_ingestion(matching_file, redis_client)
            val = redis_client.get(redis_key)
            if val:
                return json.loads(val)
        except Exception as e:
            logger.error(f"Self-healing ingestion failed for '{matching_file}': {e}")
            
    logger.error(f"Could not retrieve or ingest reference data for disease: {disease_name}")
    return []

def generate_radiology_report(disease_name: str, patient_metadata: dict, llm_model: str = "gemini-3.5-flash", for_patient: bool = False) -> str:
    """
    Main query pipeline for the RAG agent.
    Checks cache, runs vector similarity search if needed, calls LLM, and formats placeholders.
    """
    # 1. Normalize parameters
    disease_clean = disease_name.lower().strip().replace(" ", "_")
    age = get_patient_age(patient_metadata)
    age_bucket = get_age_bucket(age)
    sex = str(patient_metadata.get("gender", patient_metadata.get("sex", "male"))).lower().strip()
    if sex.startswith("m"):
        sex = "male"
    elif sex.startswith("f"):
        sex = "female"
    else:
        sex = "other"
        
    patient_name = patient_metadata.get("name", "Unknown Patient")
    current_date = patient_metadata.get("date", patient_metadata.get("studyDate"))
    if current_date:
        try:
            # Format input ISO date nicely (e.g. 2026-06-21T00:00:00.000Z -> June 21, 2026)
            parsed_date = datetime.strptime(current_date.split("T")[0], "%Y-%m-%d")
            current_date = parsed_date.strftime("%B %d, %Y")
        except Exception:
            pass
    else:
        current_date = datetime.now().strftime("%B %d, %Y")
        
    redis_client = get_redis_client()
    cache_key = f"patient_report:{disease_clean}:{age_bucket}:{sex}" if for_patient else f"rad_report:{disease_clean}:{age_bucket}:{sex}"
    
    # 2. Cache Check
    cached_template = redis_client.get(cache_key)
    if cached_template:
        logger.info(f"Cache HIT for report template: {cache_key}")
        # Perform dynamic replacements of name and date
        report_text = cached_template.decode("utf-8") if isinstance(cached_template, bytes) else str(cached_template)
        report_text = report_text.replace("{name}", patient_name)
        report_text = report_text.replace("{date}", current_date)
        report_text = report_text.replace("{age}", str(age))
        report_text = report_text.replace("{sex}", sex.capitalize())
        return report_text
        
    logger.info(f"Cache MISS for report template: {cache_key}. Proceeding to RAG similarity search...")
    
    # 3. RAG Query (Retrieve top chunks)
    chunks = fetch_document_chunks(disease_clean, redis_client)
    if not chunks:
        # Fallback if no reference data is found
        return f"Error: No reference documentation found for disease '{disease_name}' to generate report."
        
    # Embed the query (disease_name + symptoms)
    query_text = disease_name
    symptoms = patient_metadata.get("symptoms", "")
    if symptoms:
        query_text += f" presenting with symptoms: {symptoms}"
        
    response = call_gemini_with_retry(
        genai.embed_content,
        model="models/gemini-embedding-001",
        content=query_text
    )
    query_embedding = response.get('embedding', [])
    
    # Compute similarity and select top 5 chunks
    scored_chunks = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append((score, chunk))
        
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored_chunks[:5]]
    
    # 4. Construct System Prompt & User Context
    if for_patient:
        system_prompt = f"""You are an empathetic, expert clinical assistant helping translate complex radiology reports into patient-friendly language.
Your task is to generate a Patient-Friendly MRI Report based on the provided reference medical context.

Please follow this structure. Do NOT deviate from this layout:
---
PATIENT-FRIENDLY MRI SUMMARY

Dear {{name}},

This is a simplified summary of your brain MRI scan performed on {{date}}.

WHAT WAS FOUND:
[Explain in simple, gentle, clear terms what disease/anomaly was detected ({disease_name}), using friendly and non-alarmist analogies or everyday language. Avoid complex medical jargon, or explain it immediately if you must use it.]

EXPLANATION OF TECH:
[Explain in simple terms how the scan was done and why this sequence was used.]

DETAILED EXPLANATION:
[Provide a clear, easy-to-understand breakdown of what the imaging showed based on the medical context, reassuring the patient where appropriate.]

NEXT STEPS & RECOMMENDATIONS:
[List clear, actionable advice: follow-up doctor consultations, rest, or standard next steps in plain English.]

If you have any questions, please consult Dr. Tarun Ahuja, MD.
---

CRITICAL INSTRUCTIONS:
1. You MUST output the placeholders '{{name}}' and '{{date}}' exactly as written in curly braces (with no spaces inside). Do NOT replace them with actual names or dates.
2. The values for '{{age}}' and '{{sex}}' MUST be filled in with the patient's actual age (e.g. {age}) and sex (e.g. {sex.capitalize()}).
3. Write in an empathetic, supportive, and clear tone. Make sure it is completely understandable to someone without any medical background.
4. Output ONLY the report inside the '---' borders. Do not write any conversational preamble or postscript.
"""
    else:
        system_prompt = f"""You are an expert clinical radiologist assistant. Your task is to generate a standardized radiology report based on the provided reference medical context.

You MUST strictly adhere to the following format. Do NOT deviate from this layout.

---
RADIOLOGY REPORT

Patient: {{name}} | Age: {{age}} | Sex: {{sex}}
Date: {{date}}

CLINICAL INDICATION
[Detailed description of clinical indication based on symptoms, disease, and age/sex]

TECHNIQUE
[Standard imaging modality and sequences for this disease based on the reference context]

FINDINGS
[Detailed organ-by-organ anatomical and signal observations based on the reference context]

IMPRESSION
1. [Primary diagnosis/pathology: {disease_name}]
2. [Second differential diagnosis]
3. [Third differential diagnosis if applicable]

RECOMMENDATION
[Suggested follow-up scans, referral, or clinical next steps]
---

CRITICAL INSTRUCTIONS:
1. You MUST output the placeholders '{{name}}' and '{{date}}' exactly as written in curly braces (with no spaces inside). Do NOT replace them with actual names or dates.
2. The values for '{{age}}' and '{{sex}}' MUST be filled in with the patient's actual age (e.g. {age}) and sex (e.g. {sex.capitalize()}).
3. Use the provided medical reference chunks to populate the CLINICAL INDICATION, TECHNIQUE, FINDINGS, IMPRESSION, and RECOMMENDATION sections. Make the report detailed, professional, and clinically accurate.
4. Output ONLY the report inside the '---' borders. Do not write any conversational preamble or postscript.
"""

    context_str = "\n\n".join([
        f"### Section: {c['section']}\nSource: {c['source']}\nContent: {c['content']}"
        for c in top_chunks
    ])
    
    user_content = f"Medical Reference Context Chunks:\n\n{context_str}\n\nPatient Clinical Details:\n- Diagnosis: {disease_name}\n- Symptoms: {symptoms or 'N/A'}\n- Age: {age}\n- Sex: {sex.capitalize()}"
    
    # 5. Invoke LLM (trying the requested model, with fallbacks if needed)
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash-exp",
        "gemini-pro"
    ]
    if llm_model and llm_model not in models_to_try:
        models_to_try.insert(0, llm_model)
        
    llm_report_template = ""
    last_err = None
    
    for m in models_to_try:
        try:
            logger.info(f"Invoking Gemini model '{m}' for report generation...")
            model = genai.GenerativeModel(
                model_name=m,
                system_instruction=system_prompt
            )
            response = call_gemini_with_retry(model.generate_content, user_content)
            llm_report_template = response.text
            if llm_report_template:
                break
        except Exception as e:
            logger.warning(f"LLM generation failed with model '{m}': {e}")
            last_err = e
            
    if not llm_report_template:
        raise last_err or RuntimeError("Failed to generate radiology report with any Gemini model")
        
    # Clean up output formatting if LLM wrapped in code block
    llm_report_template = llm_report_template.strip()
    if llm_report_template.startswith("```"):
        llm_report_template = re.sub(r'^```[a-zA-Z]*\n', '', llm_report_template)
        llm_report_template = re.sub(r'\n```$', '', llm_report_template).strip()
        
    # 6. Cache the generated template in Upstash Redis
    redis_client.set(cache_key, llm_report_template)
    logger.info(f"Cached generated report template under key: {cache_key}")
    
    # 7. Perform final replacements and return
    report_text = llm_report_template.replace("{name}", patient_name)
    report_text = report_text.replace("{date}", current_date)
    report_text = report_text.replace("{age}", str(age))
    report_text = report_text.replace("{sex}", sex.capitalize())
    
    return report_text
