import tabula
import pandas as pd
import os
from openai import OpenAI
import json
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class InvoiceDataExtractor:
    def __init__(self, api_key: str):
        """
        Initialize the InvoiceDataExtractor with OpenAI API key.
        
        Args:
            api_key (str): OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.target_fields = ['invoice_number', 'date', 'amount']
        
    def preprocess_date(self, date_str: str) -> str:
        """
        Clean and standardize date strings, removing extra text.
        
        Args:
            date_str (str): Raw date string (e.g., "Invoice 03/06/2024" or "11/01/2024 BI")
            
        Returns:
            str: Cleaned date string in YYYY-MM-DD format or original if no date found
        """
        try:
            # Handle empty or null values
            if pd.isna(date_str) or not str(date_str).strip():
                return ""

            # Split by space and try each part to find the date
            parts = str(date_str).strip().split()
            
            # Common date formats to try
            date_formats = [
                "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y",
                "%d-%m-%Y", "%Y-%m-%d", "%m-%d-%Y",
                "%d.%m.%Y", "%Y.%m.%d",
                "%b %d %Y", "%B %d %Y",
                "%d %b %Y", "%d %B %Y"
            ]
            
            # Try each part of the string
            for part in parts:
                # Try each date format for this part
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(part, fmt)
                        return parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
            
            print(f"Warning: Could not find valid date in: {date_str}")
            return date_str
            
        except Exception as e:
            print(f"Error preprocessing date {date_str}: {str(e)}")
            return date_str
    
    def preprocess_amount(self, amount_str: str) -> str:
        """
        Clean and standardize amount strings.
        
        Args:
            amount_str (str): Raw amount string
            
        Returns:
            str: Cleaned amount string (numeric only with 2 decimal places)
        """
        try:
            # Remove currency symbols and other non-numeric characters
            amount_str = re.sub(r'[^\d.-]', '', str(amount_str))
            
            # Convert to float and format to 2 decimal places
            amount = float(amount_str)
            return f"{amount:.2f}"
            
        except Exception as e:
            print(f"Error preprocessing amount {amount_str}: {str(e)}")
            return amount_str
    
    def preprocess_invoice_number(self, invoice_str: str) -> str:
        """
        Clean and standardize invoice numbers.
        
        Args:
            invoice_str (str): Raw invoice number string
            
        Returns:
            str: Cleaned invoice number string
        """
        try:
            # Remove common prefixes
            prefixes = ['INV', 'INV#', 'INVOICE', 'INVOICE#', '#']
            result = invoice_str
            for prefix in prefixes:
                result = re.sub(f'^{prefix}\\s*', '', result, flags=re.IGNORECASE)
            
            return result.strip()
            
        except Exception as e:
            print(f"Error preprocessing invoice number {invoice_str}: {str(e)}")
            return invoice_str
    
    def analyze_sample_rows(self, table: pd.DataFrame, sample_rows: int = 3) -> Dict[str, int]:
        """
        Analyze a sample of rows to identify column patterns.
        
        Args:
            table (pd.DataFrame): DataFrame containing the table data
            sample_rows (int): Number of rows to analyze
            
        Returns:
            Dict[str, int]: Mapping of target fields to column indices
        """
        print("\nAnalyzing sample rows to identify column patterns...")
        
        # Get sample rows (skip header row)
        start_row = 1
        end_row = min(start_row + sample_rows, len(table))
        sample_data = table.iloc[start_row:end_row]
        
        # Convert sample rows to string format
        sample_str = "\n".join([
            " | ".join([f"Column {i}: {val}" for i, val in enumerate(row)])
            for _, row in sample_data.iterrows()
        ])
        
        try:
            prompt = f"""
            Analyze these rows and identify which column indices (0-based) contain these fields:
            - invoice_number
            - date
            - amount

            Data rows:
            {sample_str}

            Return ONLY a JSON object with field names as keys and column indices as integer values.
            Only include fields you are confident about. Example:
            {{"invoice_number": 2, "date": 0, "amount": 4}}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            print(f"Identified column mappings: {json.dumps(result, indent=2)}")
            return result
            
        except Exception as e:
            print(f"Error in sample analysis: {str(e)}")
            return {}
    
    def preprocess_field(self, field_name: str, value: str) -> str:
        """
        Preprocess a field based on its type.
        
        Args:
            field_name (str): Name of the field
            value (str): Value to preprocess
            
        Returns:
            str: Preprocessed value
        """
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
        """
        Extract data from table using identified column mappings.
        
        Args:
            table (pd.DataFrame): DataFrame containing the table data
            column_mappings (Dict[str, int]): Mapping of fields to column indices
            
        Returns:
            List[Dict]: Extracted data records
        """
        print("\nExtracting and preprocessing data using identified column mappings...")
        extracted_data = []
        
        # Skip header row
        for idx, row in table.iloc[1:].iterrows():
            record = {}
            for field, col_idx in column_mappings.items():
                try:
                    value = row.iloc[col_idx]
                    processed_value = self.preprocess_field(field, value)
                    if processed_value:  # Only include non-empty values
                        record[field] = processed_value
                except IndexError:
                    continue
                    
            if record:
                extracted_data.append(record)
                
        print(f"Extracted and preprocessed {len(extracted_data)} records from table")
        return extracted_data
    
    def process_table(self, table: pd.DataFrame) -> List[Dict]:
        """
        Process a single table and extract invoice information.
        
        Args:
            table (pd.DataFrame): DataFrame containing the table data
            
        Returns:
            List[Dict]: List of dictionaries containing extracted information
        """
        # Clean the table
        table = table.replace({'\r': ' ', '\n': ' '}, regex=True)
        
        # Get column mappings from sample rows
        column_mappings = self.analyze_sample_rows(table)
        
        if not column_mappings:
            print("No valid column mappings found in table")
            return []
            
        # Extract data using the identified mappings
        return self.extract_data_using_mappings(table, column_mappings)
    
    def extract_from_pdf(self, pdf_path: str, output_path: str):
        """
        Extract invoice information from PDF and save to CSV.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_path (str): Path for the output CSV file
        """
        try:
            print(f"\nProcessing PDF: {pdf_path}")
            
            # Extract tables from PDF
            print("Extracting tables from PDF...")
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            if not tables:
                print("No tables found in the PDF.")
                return
                
            print(f"Found {len(tables)} tables in the PDF")
            
            # Process each table
            all_extracted_data = []
            for i, table in enumerate(tables, 1):
                print(f"\nProcessing table {i}/{len(tables)}...")
                extracted_data = self.process_table(table)
                all_extracted_data.extend(extracted_data)
            
            # Convert to DataFrame and save
            if all_extracted_data:
                df = pd.DataFrame(all_extracted_data)
                
                # Final validation of data types
                print("\nValidating and formatting final data...")
                try:
                    if 'amount' in df.columns:
                        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"Warning during final validation: {str(e)}")
                
                df.to_csv(output_path, index=False)
                print(f"\nSuccess! Extracted {len(all_extracted_data)} records saved to {output_path}")
                print("\nSample of processed data:")
                print(df.head().to_string())
            else:
                print("No invoice data found in the tables.")
                
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")

def main():
    # Your OpenAI API key
    api_key = "your_openai_api"
    
    # Initialize extractor
    extractor = InvoiceDataExtractor(api_key)
    
    # PDF and output paths
    pdf_path = "example.pdf"
    output_path = "extracted_invoice_data.csv"
    
    # Process the PDF
    extractor.extract_from_pdf(pdf_path, output_path)

if __name__ == "__main__":
    main()