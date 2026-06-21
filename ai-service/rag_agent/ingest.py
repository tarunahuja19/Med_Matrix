import os
import re
import sys
import google.generativeai as genai
from rag_agent.config import get_gemini_api_key
from rag_agent.store import store_disease_chunks, clear_all_documents

def parse_markdown_sections(text: str) -> list:
    """Parses a disease reference markdown file into individual section chunks."""
    # Split the file by Markdown H2 headers (e.g. ## 1. Section Name)
    pattern = r'(^##\s+.*$)'
    parts = re.split(pattern, text, flags=re.MULTILINE)
    
    # Content before the first H2 is the Overview
    overview_content = parts[0].strip()
    
    sections = []
    if overview_content:
        # Strip the H1 header from overview content if present to keep it clean
        overview_clean = re.sub(r'^#\s+.*$', '', overview_content, flags=re.MULTILINE).strip()
        if overview_clean:
            sections.append({
                "section": "Overview",
                "content": overview_clean
            })
            
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        
        # Clean heading, e.g. "## 2. Radiographic & Imaging Findings" -> "Radiographic & Imaging Findings"
        section_name = re.sub(r'^##\s*(\d+\.\s*)?', '', heading).strip()
        
        if content:
            sections.append({
                "section": section_name,
                "content": content
            })
            
    return sections

def ingest_disease_file(file_path: str, source_name: str = "MedMatrix Disease Reference") -> bool:
    """Reads a single markdown disease reference file, generates embeddings with Gemini, and saves to Upstash Redis."""
    api_key = get_gemini_api_key()
    if not api_key:
        print("❌ Error: No GEMINI_API_KEYS are configured in .env.")
        return False
        
    filename = os.path.basename(file_path)
    disease_name = os.path.splitext(filename)[0]
    
    print(f"Processing '{disease_name}' from {filename}...")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        sections = parse_markdown_sections(text)
        if not sections:
            print(f"⚠️ No sections found in {filename}.")
            return False
            
        print(f"Found {len(sections)} sections. Generating Gemini embeddings...")
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Prepare all contents for batch embedding
        contents = [sec["content"] for sec in sections]
        cleaned_contents = [c.replace("\n", " ") for c in contents]
        
        # Call Gemini Embeddings API (batch request)
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=cleaned_contents
        )
        
        # Extract embeddings list from response
        embeddings = response.get("embedding", [])
        if not embeddings:
            print(f"❌ Failed to get embeddings from Gemini API for {disease_name}")
            return False
            
        # Store all sections in Upstash Redis as a JSON list
        chunks_to_store = []
        for idx, sec in enumerate(sections):
            embedding = embeddings[idx]
            chunks_to_store.append({
                "section": sec["section"],
                "content": sec["content"],
                "source": source_name,
                "embedding": embedding
            })
            
        success = store_disease_chunks(disease_name, chunks_to_store)
        if success:
            print(f"Successfully stored {len(sections)} chunks in Redis for {disease_name}.")
        return success
        
    except Exception as e:
        print(f"❌ Failed to ingest {filename}: {e}", file=sys.stderr)
        return False

def run_full_ingestion(data_dir: str) -> int:
    """Scans data directory and ingests all Markdown files found."""
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        return 0
        
    md_files = [f for f in os.listdir(data_dir) if f.endswith(".md") and f != "hi,.md"]
    if not md_files:
        print("⚠️ No Markdown files found to ingest.")
        return 0
        
    print(f"Found {len(md_files)} reference files to ingest.")
    
    # Clear existing documents in Upstash Redis
    clear_all_documents()
    
    successful_files = 0
    for filename in md_files:
        file_path = os.path.join(data_dir, filename)
        if ingest_disease_file(file_path):
            successful_files += 1
            
    print(f"\n🎉 Ingestion complete! Successfully processed {successful_files}/{len(md_files)} files in Redis.")
    return successful_files

if __name__ == "__main__":
    default_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    run_full_ingestion(default_data_dir)
