# Section 1: System Imports & Core Framework Initialization
import os
import json
import sqlite3
from typing import List
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image as PILImage
from google import genai
from dotenv import load_dotenv

# Activate environment configuration parser
load_dotenv()

# Initialize core server application instance
app = FastAPI()

# Link the dynamic HTML templates workspace directory
templates = Jinja2Templates(directory="templates")

# Bootstrap the modern Google GenAI high-performance client engine
client = genai.Client()

# Set local database storage target filename
DB_FILE = "ocr_records.db"


# Section 2: Lightweight Serverless Database Schema Management
def init_db():
    """Initializes a brand new tracking database with auto-increment sequences."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stored_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            extracted_json TEXT
        )
    """)
    conn.commit()
    conn.close()

# Fire database verification setup sequence immediately upon server boot
init_db()


# Section 3: Home Console Application Routing Gateway (GET /)
@app.get("/", response_class=HTMLResponse)
async def ui_homepage(request: Request):
    """Queries all historical clinical log files sorted with newest entries first."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.execute("SELECT id, filename, extracted_json FROM stored_documents ORDER BY id DESC")
    
    # Stream the pre-indented database string directly into raw_json
    history = [{"id": r[0], "filename": r[1], "data": json.loads(r[2]), "raw_json": r[2]} for r in cursor.fetchall()]
    conn.close()
    
    return templates.TemplateResponse(request, "index.html", {
        "history_records": history,
        "discovered_keys": None,
        "bulk_results": None
    })


# Section 4: Phase 1 Route — Multiple Template Layout Auto-Discovery
@app.post("/discover-keys", response_class=HTMLResponse)
async def ui_discover_keys(request: Request, image_files: List[UploadFile] = File(...)):
    """Scans structural anchors across multiple empty template layout files and merges them."""
    
    discovery_prompt = """
    Analyze this medical document template image. Identify all form field labels, question metrics, 
    or database column keys printed on it (e.g., 'Patient Name', 'Age', 'BPM', 'Blood Group', 'Symptoms'). 
    Do NOT extract data inputs or handwriting entries. Return ONLY a valid JSON object containing 
    a clean list of these field labels under a 'discovered_keys' property key.
    
    Format target:
    {
        "discovered_keys": ["Patient Name", "Age", "Blood Group"]
    }
    """
    
    unique_keys = set()
    
    for upload_item in image_files:
        if not upload_item.filename:
            continue
            
        try:
            img = PILImage.open(upload_item.file)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img, discovery_prompt]
            )
            
            raw_response = response.text.strip()
            if raw_response.startswith("```"):
                raw_response = raw_response.strip("`").strip()
            if raw_response.lower().startswith("json"):
                raw_response = raw_response[4:].strip()
                
            payload = json.loads(raw_response)
            file_keys = payload.get("discovered_keys", [])
            
            for key in file_keys:
                unique_keys.add(key)
                
        except Exception as e:
            unique_keys.add(f"Discovery Fault ({upload_item.filename}): {str(e)}")

    discovered_keys = sorted(list(unique_keys))

    # Refresh history records for UI tracking consistency
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.execute("SELECT id, filename, extracted_json FROM stored_documents ORDER BY id DESC")
    history = [{"id": r[0], "filename": r[1], "data": json.loads(r[2]), "raw_json": r[2]} for r in cursor.fetchall()]
    conn.close()

    return templates.TemplateResponse(request, "index.html", {
        "history_records": history,
        "discovered_keys": discovered_keys,
        "bulk_results": None
    })


# Section 5: Phase 2 Route — Multiple Images Batch Matching with Exclusion Filters
@app.post("/extract-targeted-values", response_class=HTMLResponse)
async def ui_extract_values(
    request: Request, 
    selected_keys: List[str] = Form(...), 
    value_images: List[UploadFile] = File(...)
):
    """Loops through image files, extracts values, and filters out non-applicable keys."""
    bulk_results = {}
    
    keys_structure = ", ".join([f'"{k}": "extracted value"' for k in selected_keys])
    extraction_prompt = f"""
    Analyze this filled clinical document image. 
    You are provided with a strict validation checklist array: {selected_keys}.
    Locate each target key from that list inside the image and extract its written data entry value.
    Return ONLY a valid JSON object containing these custom mappings. If a field value is blank or missing, write 'N/A'.

    Format target:
    {{
        {keys_structure}
    }}
    """
    
    for upload_item in value_images:
        if not upload_item.filename:
            continue
            
        current_filename = upload_item.filename
        
        try:
            img = PILImage.open(upload_item.file)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img, extraction_prompt]
            )
            
            raw_response = response.text.strip()
            if raw_response.startswith("```"):
                raw_response = raw_response.strip("`").strip()
            if raw_response.lower().startswith("json"):
                raw_response = raw_response[4:].strip()
                
            parsed_answers = json.loads(raw_response)
            
            # 🛠️ DATA SANITIZATION FILTER:
            # Drop any keys where the value is 'N/A' or empty to prevent cross-image pollution
            clean_answers = {
                k: v for k, v in parsed_answers.items() 
                if v and str(v).strip().upper() != "N/A"
            }
            
            # Only track the file results if it contains valid extractions
            if clean_answers:
                bulk_results[current_filename] = clean_answers
            
        except Exception as e:
            bulk_results[current_filename] = {"Batch Engine Fault": str(e)}

    # Save the consolidated clean data package into one row entry if entries exist
    if bulk_results:
        combined_filenames = ", ".join(bulk_results.keys())
        
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            "INSERT INTO stored_documents (filename, extracted_json) VALUES (?, ?)",
            (combined_filenames, json.dumps(bulk_results, indent=2))
        )
        conn.commit()
        conn.close()

    # Re-query global history logs
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.execute("SELECT id, filename, extracted_json FROM stored_documents ORDER BY id DESC")
    history = [{"id": r[0], "filename": r[1], "data": json.loads(r[2]), "raw_json": r[2]} for r in cursor.fetchall()]
    conn.close()

    return templates.TemplateResponse(request, "index.html", {
        "history_records": history,
        "discovered_keys": None,
        "bulk_results": bulk_results 
    })


# Section 6: Clean Record Trash Purging Interface Channel
@app.post("/delete-record/{record_id}")
async def ui_delete_record(record_id: int):
    """Drops a history record entry log cleanly using relational unique index parameter keys."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM stored_documents WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return RedirectResponse(url="/", status_code=303)