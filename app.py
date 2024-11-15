import fitz
import os
import json
import re
import pandas as pd
import openai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from datetime import datetime
import logging
from typing import List, Dict, Optional
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="PDF Data Extractor🤖",
    description="Extract structured data from PDF files",
    version="0.1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load OpenAI API key from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

class DataProcessor:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.ensure_cache_dir()
        openai.api_key = api_key
        
        # Compile regex patterns for pre-processing
        self.patterns = {
            'reference': r'(?i)(?:ref(?:erence)?(?:\s)?(?:no|number|#)?[\s:]*)([A-Z0-9-]+)',
            'date': r'(?i)(?:date[d]?[\s:]*)(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2})',
            'amount': r'(?i)(?:amount|total|sum)[\s:]*[$€£]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        }
        self.compiled_patterns = {
            key: re.compile(pattern) for key, pattern in self.patterns.items()
        }

    def ensure_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if time.time() - cached_data['timestamp'] < 86400:
                        return cached_data['data']
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None

    def save_to_cache(self, cache_key: str, data: List[Dict]):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'timestamp': time.time(), 'data': data}, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def pre_process_chunk(self, chunk: str) -> List[Dict]:
        results = []
        lines = chunk.split('\n')
        for i in range(len(lines)):
            line_context = ' '.join(lines[max(0, i-2):min(len(lines), i+3)])
            matches = {field: self.compiled_patterns[field].search(line_context) for field in self.patterns}
            if all(matches.values()):
                results.append({
                    'Reference number': matches['reference'].group(1),
                    'Date': matches['date'].group(1),
                    'Amount': matches['amount'].group(1)
                })
        return results

    def validate_data(self, data: List[Dict]) -> List[Dict]:
        validated = []
        for entry in data:
            try:
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y']:
                    try:
                        date_obj = datetime.strptime(entry['Date'], fmt)
                        entry['Date'] = date_obj.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
                entry['Amount'] = re.sub(r'[^0-9.]', '', entry['Amount'])
                float(entry['Amount'])
                validated.append(entry)
            except Exception as e:
                logger.warning(f"Validation error: {e}")
        return validated

    def process_chunk_with_gpt(self, chunk: str) -> List[Dict]:
        cache_key = self.get_cache_key(chunk)
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        regex_results = self.pre_process_chunk(chunk)
        if regex_results:
            validated_results = self.validate_data(regex_results)
            if validated_results:
                self.save_to_cache(cache_key, validated_results)
                return validated_results

        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = (
                    "Extract Reference number, Date, and Amount from this text. Rules:\n"
                    "1. Only include entries with ALL fields present.\n"
                    "2. Skip partial matches.\n"
                    "3. Ensure fields are related.\n"
                    "Return as a JSON array with keys: 'Reference number', 'Date', 'Amount'.\n"
                    f"Text:\n{chunk}\n"
                )
                client = openai.Client(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.0
                )

                content = response.choices[0].message.content
                json_match = re.search(r'\[.*\]', content, re.DOTALL)

                if json_match:
                    data = json.loads(json_match.group(0))
                    validated_data = self.validate_data(data)
                    if validated_data:
                        self.save_to_cache(cache_key, validated_data)
                        return validated_data

                return []

            except Exception as e:
                logger.warning(f"GPT processing attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)

        return []

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        extracted_text = []
        for page in doc:
            extracted_text.append(page.get_text())
        doc.close()
        full_text = '\n'.join(extracted_text)
        chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            chunk_results = list(executor.map(self.process_chunk_with_gpt, chunks))
        for result in chunk_results:
            results.extend(result)

        return results

processor = DataProcessor()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

        pdf_path = f"temp_{file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        data = processor.process_pdf(pdf_path)
        os.remove(pdf_path)

        if not data:
            return JSONResponse(content={"message": "No valid data extracted"}, status_code=200)

        return JSONResponse(content={"data": data}, status_code=200)

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
