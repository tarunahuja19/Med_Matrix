import os
import re
import json
import logging
from upstash_redis import Redis
import google.generativeai as genai
from rag_agent.gemini_client import call_gemini_with_retry

logger = logging.getLogger("rag-agent-ingest")

def get_redis_client():
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if not url or not token:
        raise ValueError("Upstash Redis credentials (UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN) missing in environment")
    return Redis(url=url, token=token)

def parse_markdown_file(file_path):
    """
    Parses a disease reference markdown file, splitting it by H2 headers
    to extract sections with their titles and contents.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by H2 heading (starting with ## at start of line or following newline)
    parts = re.split(r'\n##\s+', '\n' + text)
    sections = []
    
    for part in parts[1:]:
        lines = part.split('\n')
        if not lines:
            continue
        title = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()
        if title and content:
            sections.append({
                "section": title,
                "content": content,
                "source": "MedMatrix Disease Reference"
            })
            
    return sections

def embed_sections(sections):
    """
    Batch embeds all sections using Gemini text-embedding-004.
    """
    if not sections:
        return []
        
    contents = [f"{sec['section']}\n\n{sec['content']}" for sec in sections]
    
    response = call_gemini_with_retry(
        genai.embed_content,
        model="models/gemini-embedding-001",
        content=contents
    )
    
    embeddings = response.get('embedding', [])
    for i, emb in enumerate(embeddings):
        sections[i]["embedding"] = emb
        
    return sections

def run_single_file_ingestion(file_path, redis_client=None):
    """Parses, embeds, and stores a single markdown file into Upstash Redis."""
    if redis_client is None:
        redis_client = get_redis_client()
        
    basename = os.path.basename(file_path)
    disease_name = os.path.splitext(basename)[0].lower()
    
    logger.info(f"Ingesting file: {file_path} for disease: {disease_name}...")
    sections = parse_markdown_file(file_path)
    if not sections:
        logger.warning(f"No sections parsed from {file_path}")
        return 0
        
    sections_with_embeddings = embed_sections(sections)
    
    redis_key = f"med_docs:{disease_name}"
    # Serialize to JSON and save to Upstash Redis
    redis_client.set(redis_key, json.dumps(sections_with_embeddings))
    logger.info(f"Successfully stored {len(sections_with_embeddings)} chunks under Redis key: {redis_key}")
    return len(sections_with_embeddings)

def run_full_ingestion(data_dir):
    """Ingests all markdown files from the specified data directory."""
    logger.info(f"Starting full RAG ingestion from directory: {data_dir}...")
    redis_client = get_redis_client()
    
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist")
        
    md_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.md')
    ]
    
    if not md_files:
        logger.warning(f"No markdown reference files found in {data_dir}")
        return 0
        
    success_count = 0
    for file_path in md_files:
        try:
            run_single_file_ingestion(file_path, redis_client)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            
    logger.info(f"Full ingestion completed. Successfully processed {success_count}/{len(md_files)} files.")
    return success_count
