import tabula
import pandas as pd
import os
from openai import OpenAI
import json
from typing import Dict, List, Tuple
import re
from datetime import datetime
import warnings
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize FastAPI application
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceDataExtractor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.target_fields = ['invoice_number', 'date', 'amount']
        
    def preprocess_date(self, date_str: str) -> str:
        """Convert various date formats to a standardized format."""
        try:
            if pd.isna(date_str) or not str(date_str).strip():
                return ""
            date_formats = [
                "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y",
                "%d-%m-%Y", "%Y-%m-%d", "%m-%d-%Y",
                "%d.%m.%Y", "%Y.%m.%d", "%b %d %Y",
                "%B %d %Y", "%d %b %Y", "%d %B %Y"
            ]
            for part in str(date_str).strip().split():
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(part, fmt)
                        return parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
            return date_str
        except Exception as e:
            logger.error(f"Error preprocessing date {date_str}: {e}")
            return date_str
    
    def preprocess_amount(self, amount_str: str) -> str:
        """Convert amount to a clean decimal format."""
        try:
            amount_str = re.sub(r'[^\d.-]', '', str(amount_str))
            amount = float(amount_str)
            return f"{amount:.2f}"
        except Exception as e:
            logger.error(f"Error preprocessing amount {amount_str}: {e}")
            return amount_str
    
    def preprocess_invoice_number(self, invoice_str: str) -> str:
        """Remove common prefixes from invoice numbers."""
        try:
            prefixes = ['INV', 'INV#', 'INVOICE', 'INVOICE#', '#']
            result = invoice_str
            for prefix in prefixes:
                result = re.sub(f'^{prefix}\\s*', '', result, flags=re.IGNORECASE)
            return result.strip()
        except Exception as e:
            logger.error(f"Error preprocessing invoice number {invoice_str}: {e}")
            return invoice_str

    def analyze_sample_rows(self, table: pd.DataFrame, sample_rows: int = 3) -> Tuple[Dict[str, int], bool]:
        """Analyze table structure to identify column mappings."""
        try:
            if table.empty:
                logger.info("Empty table detected. Skipping analysis.")
                return {}, False

            analysis_str = "Column Names (potential header row):\n" + " | ".join([f"Column {i}: {col}" for i, col in enumerate(table.columns)])
            for idx, row in table.head(sample_rows).iterrows():
                analysis_str += " | ".join([f"Column {i}: {val}" for i, val in enumerate(row)]) + "\n"

            prompt = f"""
            Analyze this table structure to:
            1. Determine if the column names row is actually a data row that tabula incorrectly identified as headers.
            2. Identify which columns contain these fields:
            - invoice_number (unique identifier in the 'Num' column, e.g., 'RC24-4423')
            - date (a valid date like 'DD/MM/YYYY' or 'YYYY-MM-DD')
            - amount (a valid numeric value)
            Table Structure:
            {analysis_str}
            Example response:
            {{"column_mappings": {{"invoice_number": 1, "date": 0, "amount": 7}}, "include_header_as_data": true}}
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a precise data extraction assistant. Respond only with valid JSON."},
                          {"role": "user", "content": prompt}],
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)
            return result['column_mappings'], result['include_header_as_data']
        except Exception as e:
            logger.error(f"Error in sample analysis: {e}")
            return {}, False

    def is_valid_record(self, record: Dict) -> bool:
        """Check if all required fields in a record have non-empty values."""
        return all(record.get(field, "").strip() != "" for field in self.target_fields)

    def preprocess_field(self, field_name: str, value: str) -> str:
        """Preprocess fields based on their type."""
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
    
    def extract_data_using_mappings(self, table: pd.DataFrame, column_mappings: Dict[str, int], include_header_as_data: bool) -> List[Dict]:
        """Extract data from table using specified column mappings."""
        extracted_data = []
        if include_header_as_data:
            header_row = {field: self.preprocess_field(field, table.columns[col_idx]) for field, col_idx in column_mappings.items()}
            if self.is_valid_record(header_row):
                extracted_data.append(header_row)
        
        for _, row in table.iterrows():
            record = {field: self.preprocess_field(field, row.iloc[col_idx]) for field, col_idx in column_mappings.items()}
            if self.is_valid_record(record):
                extracted_data.append(record)
        
        return extracted_data
    
    def process_table(self, table: pd.DataFrame) -> List[Dict]:
        """Process each table by analyzing structure and extracting relevant data."""
        table.replace({'\r': ' ', '\n': ' '}, regex=True, inplace=True)
        column_mappings, include_header_as_data = self.analyze_sample_rows(table)
        return self.extract_data_using_mappings(table, column_mappings, include_header_as_data) if column_mappings else []
    
    def extract_from_pdf(self, pdf_path: str, output_path: str):
        """Extract data from each table in the PDF and save results."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            if not tables:
                logger.warning("No tables found in the PDF.")
                return []

            all_extracted_data = pd.DataFrame()
            for i, table in enumerate(tables, 1):
                extracted_data = self.process_table(table)
                if extracted_data:
                    all_extracted_data = pd.concat([all_extracted_data, pd.DataFrame(extracted_data)], ignore_index=True)

            if not all_extracted_data.empty:
                all_extracted_data.to_csv(output_path, index=False)
                logger.info(f"Success! Extracted data saved to {output_path}")
                return all_extracted_data.to_dict(orient='records')
            else:
                logger.info("No valid invoice data found in the tables.")
                return []
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return []

@app.post("/extract_invoice_data")
async def extract_invoice_data(pdf_file: UploadFile = File(...)):
    try:
        api_key = "your_api_key"
        extractor = InvoiceDataExtractor(api_key)
        pdf_bytes = await pdf_file.read()
        pdf_path = f"{pdf_file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        output_path = "extracted_invoice_data.csv"
        extracted_data = extractor.extract_from_pdf(pdf_path, output_path)
        
        return JSONResponse(content=extracted_data)
    except Exception as e:
        logger.error(f"Error in API endpoint: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
