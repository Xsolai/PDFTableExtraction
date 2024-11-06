from fastapi import FastAPI, File, UploadFile, HTTPException
import tabula
import pandas as pd
import os
from openai import OpenAI
import json
from typing import Dict, List, Optional
import re
from datetime import datetime
import warnings
import tempfile

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Invoice Data Extractor🔰",
    description="Extract invoice data from PDF files using OpenAI's GPT-3.",
    version="0.1"
)

class InvoiceDataExtractor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.target_fields = ['invoice_number', 'date', 'amount']
        
    def preprocess_date(self, date_str: str) -> str:
        try:
            if pd.isna(date_str) or not str(date_str).strip():
                return ""
            parts = str(date_str).strip().split()
            date_formats = [
                "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y",
                "%d-%m-%Y", "%Y-%m-%d", "%m-%d-%Y",
                "%d.%m.%Y", "%Y.%m.%d",
                "%b %d %Y", "%B %d %Y",
                "%d %b %Y", "%d %B %Y"
            ]
            for part in parts:
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(part, fmt)
                        return parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
            return date_str
        except Exception as e:
            return date_str
    
    def preprocess_amount(self, amount_str: str) -> str:
        try:
            amount_str = re.sub(r'[^\d.-]', '', str(amount_str))
            amount = float(amount_str)
            return f"{amount:.2f}"
        except Exception as e:
            return amount_str
    
    def preprocess_invoice_number(self, invoice_str: str) -> str:
        try:
            prefixes = ['INV', 'INV#', 'INVOICE', 'INVOICE#', '#']
            result = invoice_str
            for prefix in prefixes:
                result = re.sub(f'^{prefix}\\s*', '', result, flags=re.IGNORECASE)
            return result.strip()
        except Exception as e:
            return invoice_str
    
    def analyze_sample_rows(self, table: pd.DataFrame, sample_rows: int = 3) -> Dict[str, int]:
        start_row = 1
        end_row = min(start_row + sample_rows, len(table))
        sample_data = table.iloc[start_row:end_row]
        sample_str = "\n".join([
            " | ".join([f"Column {i}: {val}" for i, val in enumerate(row)])
            for _, row in sample_data.iterrows()
        ])
        prompt = f"""
        Analyze these rows and identify which column indices (0-based) contain these fields:
        - invoice_number
        - date
        - amount
        Data rows:
        {sample_str}
        Return ONLY a JSON object with field names as keys and column indices as integer values.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            return {}
    
    def preprocess_field(self, field_name: str, value: str) -> str:
        if pd.isna(value) or not str(value).strip():
            return ""
        value = str(value).strip()
        if field_name == 'date':
            return self.preprocess_date(value)
        elif field_name == 'amount':
            return self.preprocess_amount(value)
        elif field_name == 'invoice_number':
            return self.preprocess_invoice_number(value)
        return value
    
    def extract_data_using_mappings(self, table: pd.DataFrame, column_mappings: Dict[str, int]) -> List[Dict]:
        extracted_data = []
        for _, row in table.iloc[1:].iterrows():
            record = {}
            for field, col_idx in column_mappings.items():
                try:
                    value = row.iloc[col_idx]
                    processed_value = self.preprocess_field(field, value)
                    if processed_value:
                        record[field] = processed_value
                except IndexError:
                    continue
            if record:
                extracted_data.append(record)
        return extracted_data
    
    def process_table(self, table: pd.DataFrame) -> List[Dict]:
        table = table.replace({'\r': ' ', '\n': ' '}, regex=True)
        column_mappings = self.analyze_sample_rows(table)
        if not column_mappings:
            return []
        return self.extract_data_using_mappings(table, column_mappings)
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        try:
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            if not tables:
                return []
            all_extracted_data = []
            for table in tables:
                extracted_data = self.process_table(table)
                all_extracted_data.extend(extracted_data)
            return all_extracted_data
        except Exception as e:
            return []

@app.post("/extract-invoice-data")
async def extract_invoice_data(file: UploadFile = File(...)):
    try:
        # Save the uploaded PDF file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(await file.read())
            temp_pdf_path = temp_pdf.name

        # Initialize the extractor with your OpenAI API key
        api_key = "your_openai_api"
        extractor = InvoiceDataExtractor(api_key)

        # Extract data from the PDF
        extracted_data = extractor.extract_from_pdf(temp_pdf_path)
        
        # Delete the temp file
        os.remove(temp_pdf_path)
        
        if not extracted_data:
            raise HTTPException(status_code=400, detail="No data extracted from the PDF.")

        return {"extracted_data": extracted_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
