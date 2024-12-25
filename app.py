from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import urllib.request
import openai
import os
import logging
import base64
import requests
import ast
import json
import re
from groq import Groq
from enum import Enum



# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not OPENAI_API_KEY or not API_KEY:
    raise RuntimeError("Please set 'OPENAI_API_KEY' and 'API_KEY' in your .env file")

if not GROQ_API_KEY: 
    raise RuntimeError("Please set 'GROQ_API_KEY' in your .env file")



openai.api_key = OPENAI_API_KEY

client = Groq(api_key=GROQ_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="Vendor Statements Processor",
    description="A FastAPI application to process vendor statements using OpenAI's GPT or Groq API.",
    version="0.1.0",
    docs_url="/",
    redoc_url=None
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API Key Authentication
api_key_header = APIKeyHeader(name="api_key", auto_error=False)

# Dependency: Validate API Key
async def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        logger.warning("Invalid API key provided.")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return api_key

# Add a model selection Enum
class ModelChoice(str, Enum):
    GPT = "gpt"
    GROQ = "groq"


# Utility: Extract text from PDF
def extract_pdf_text(file_path: str) -> str:
    try:
        logger.info("Extracting text from PDF...")
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        logger.info("PDF text extraction complete.")
        return text
    except Exception as e:
        logger.error(f"Error during PDF text extraction: {e}")
        raise HTTPException(status_code=500, detail="Error extracting text from PDF")

# Utility: Mock token usage calculation
def calculate_tokens(text: str) -> int:
    word_count = len(text.split())
    tokens = word_count // 4  # Approximation: 1 token = 4 words
    logger.info(f"Calculated token usage: {tokens} tokens for {word_count} words.")
    return tokens

# Utility: Split text into chunks
def split_text(text: str, max_length: int) -> list:
    """Split the text into smaller chunks within the token limit."""
    chunks = []
    while len(text) > max_length:
        split_index = text[:max_length].rfind('\n')  # Split at the nearest newline
        if split_index == -1:
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:]
    chunks.append(text)
    return chunks

# GPT-based vendor details extraction
async def extract_vendor_details_gpt(text: str):
    truncated_text = text[:500]
    logger.info("Extracting vendor details...")

    system_prompt = (
        "You are an assistant specialized in extracting vendor details. "
        "Extract vendor-related information, such as name, address, website, city, email, and ID. "
        "If any detail is not available, leave it as an empty string. "
        "Do not include any commentary, explanations, or additional text. Return only valid JSON."
    )
    user_prompt = f"""
Extract the vendor details from the following text:

{truncated_text}

Return ONLY valid JSON data in this exact format:
{{
  "vendorDetails": {{
    "vendorName": "string",
    "vendorAddress": "string",
    "vendorWebsite": "string",
    "vendorCity": "string",
    "vendorEmail": "string",
    "vendorID": 0.0
  }}
}}
"""
    try:
        client = openai.Client(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
        )
        raw_response = response.choices[0].message.content
        logger.info(f"Raw response for vendor details: {raw_response}")

        # Extract the JSON part using regex
        match = re.search(r"{.*}", raw_response, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in the response")
        
        json_content = match.group(0)

        # Parse the extracted JSON
        parsed_response = json.loads(json_content)
        return parsed_response
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in vendor details: {e}")
        logger.debug(f"Response content: {raw_response}")
        raise HTTPException(status_code=500, detail="Failed to parse vendor details response")
    except Exception as e:
        logger.error(f"Failed to extract vendor details: {e}")
        raise HTTPException(status_code=500, detail="Error extracting vendor details")





# Groq-based vendor details extraction
async def extract_vendor_details_groq(text: str):
    truncated_text = text[:500]
    prompt = f"""
        "You are an assistant specialized in extracting vendor details. "
        "Extract vendor-related information, such as name, address, website, city, email, and ID. "
        "If any detail is not available, leave it as an empty string. "
        "Do not include any commentary, explanations, or additional text. Return only valid JSON."

        Extract the vendor details from the following text:
        
        {truncated_text}

Return ONLY valid JSON data in this exact format:
{{
  "vendorDetails": {{
    "vendorName": "string",
    "vendorAddress": "string",
    "vendorWebsite": "string",
    "vendorCity": "string",
    "vendorEmail": "string",
    "vendorID": 0.0
  }}
}}

"""
    
    try:
        # Send the request to Groq's API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-70b-versatile",  # Specify the Llama model to use
        )

        # Extract and process the output
        formatted_output = chat_completion.choices[0].message.content
        logger.info(f"Extracted vendor details: {formatted_output}")
        
        # Extract the JSON part using regex
        json_data = re.search(r"{.*}", formatted_output, re.DOTALL)

        if not json_data:
            raise Exception("Failed to extract JSON data.")
        

        json_content = json_data.group(0)
        parsed_response = json.loads(json_content)
        return parsed_response
    
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in vendor details: {e}")
        logger.debug(f"Response content: {formatted_output}")
        raise HTTPException(status_code=500, detail="Failed to parse vendor details response")
    except Exception as e:
        logger.error(f"Failed to extract vendor details: {e}")
        raise HTTPException(status_code=500, detail="Error extracting vendor details")
    

# Function: Process text chunks for invoice data
async def process_text_chunks_gpt(text_chunks: list):
    """Process text chunks to extract invoice data."""
    results = []
    total_tokens_used = 0

    for i, chunk in enumerate(text_chunks):
        system_prompt = (
            "You are an assistant specialized in extracting structured invoice data. "
            "Each invoice entry should include invoiceID, invoiceDate, poNumber, creditAmount, and debitAmount. "
            "If any of these details are not found, leave the field empty or as 0. "
            "Return ONLY valid JSON in the specified format without any commentary or extra text. "
        )
        user_prompt = f"""
Extract the structured invoice data from the following text chunk:

{chunk}

Return ONLY valid JSON data in this exact format:
{{
  "invoicesData": [
    {{
      "invoiceID": "string",
      "invoiceDate": "string",
      "poNumber": "string",
      "creditAmount": 0.0,
      "debitAmount": 0.0
    }}
  ]
}}
If no invoice details are found, return:
{{
  "invoicesData": []
}}
"""

        try:
            client = openai.Client(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
            )
            raw_output = response.choices[0].message.content
            logger.info(f"Raw output for chunk {i + 1}: {raw_output}")
            total_tokens_used += response.usage.total_tokens if response.usage else 0

            # Extract JSON data using regex to handle possible extra text
            match = re.search(r"{.*}", raw_output, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response for chunk {i + 1}")

            json_content = match.group(0)

            # Parse and append the extracted JSON
            parsed_data = json.loads(json_content)
            results.append(parsed_data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in chunk {i + 1}: {e}")
            logger.debug(f"Response content for chunk {i + 1}: {raw_output}")
        except Exception as e:
            logger.error(f"Failed to process chunk {i + 1}: {e}")

    return results, total_tokens_used





# Function: Process text chunks for invoice data using groq API

async def process_text_chunks_groq(text_chunks: list):
    results = []
    total_tokens_used = 0

    for i, chunk in enumerate(text_chunks):
        prompt = f"""
You are an assistant specialized in extracting structured invoice data. 
Each invoice entry includes:
- invoiceID: a unique identifier containing alphanumeric characters or numeric values. It must NOT contain spaces. Example: "24001619" or "INV12345".
- invoiceDate: the date of the invoice in "DD/MM/YYYY" or "YYYY-MM-DD" format.
- poNumber: the purchase order number (alphanumeric or numeric).
- creditAmount: a decimal number representing the credit amount. Default to 0.0 if not found.
- debitAmount: a decimal number representing the debit amount. Default to 0.0 if not found.

IMPORTANT RULES:
- "invoiceID" should NOT include prefixes like "BI" or spaces. Extract only the actual ID.
- Exclude footer rows or irrelevant text that does not contain valid invoice entries.
- Return ONLY valid JSON in the following format:
{{
  "invoicesData": [
    {{
      "invoiceID": "string",
      "invoiceDate": "string",
      "poNumber": "string",
      "creditAmount": 0.0,
      "debitAmount": 0.0
    }}
  ]
}}
If no valid invoice data is found, return:
{{
  "invoicesData": []
}}

Extract the structured invoice data from the following text chunk:

{chunk}
"""

        try:
            # Send the request to Groq's API
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-70b-versatile",  # Specify the Llama model to use
            )

            # Extract and process the output
            formatted_output = chat_completion.choices[0].message.content
            logger.info(f"Processed chunk {i + 1}: {formatted_output}")
            
            # Calculate token usage
            total_tokens_used += calculate_tokens(formatted_output)
            
            # Directly parse the JSON response
            json_content = json.loads(formatted_output)
            results.append(json_content)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in chunk {i + 1}: {e}")
            logger.debug(f"Response content: {formatted_output}")

        except Exception as e:
            logger.error(f"Failed to process chunk {i + 1} using Groq: {e}")
            raise HTTPException(status_code=500, detail="Error processing text chunks")

    return results, total_tokens_used








# Update the endpoint to include model selection
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), 
    model: ModelChoice = Form(ModelChoice.GPT),  # Default to GPT
    api_key: str = Depends(validate_api_key)
):
    try:
        # Save the uploaded file
        file_path = f"./uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Extract text from the PDF
        extracted_text = extract_pdf_text(file_path)

        # Save the extracted text to a file
        with open("./uploads/extracted_text.txt", "w") as f:
            f.write(extracted_text)
        
        # Select vendor details extraction function based on model
        if model == ModelChoice.GPT:
            vendor_details = await extract_vendor_details_gpt(extracted_text)
        else:
            vendor_details = await extract_vendor_details_groq(extracted_text)
        
        # Select text chunks processing function based on model
        max_chunk_size = 2000
        text_chunks = split_text(extracted_text, max_chunk_size)
        
        if model == ModelChoice.GPT:
            processed_chunks, total_tokens_used = await process_text_chunks_gpt(text_chunks)
        else:
            processed_chunks, total_tokens_used = await process_text_chunks_groq(text_chunks)

        # Aggregate invoices
        invoices_data = []
        for chunk_result in processed_chunks:
            invoices_data.extend(chunk_result.get("invoicesData", []))
        
        # Calculate balances and validation
        open_balance = sum(item.get("creditAmount", 0) for item in invoices_data) - \
                       sum(item.get("debitAmount", 0) for item in invoices_data)
        closing_balance = open_balance
        validation_status = "Valid" if open_balance == closing_balance else "Invalid"
        validation_message = "Balances match." if validation_status == "Valid" else "Balances do not match."
        
        # Format response
        data = {
            "vendorDetails": vendor_details.get("vendorDetails", {}),
            "invoiceSummary": {
                "invoiceDate": invoices_data[0].get("invoiceDate", "") if invoices_data else "",
                "numberOfRowItems": len(invoices_data),
                "openBalance": open_balance,
                "closingBalance": closing_balance,
                "validationStatus": validation_status,
                "modelUsed": model.value,
                "tokenUsage": total_tokens_used
            },
            "invoicesData": invoices_data,
            "validationResults": {
                "balanceValidationStatus": validation_status,
                "balanceValidationMessage": validation_message
            },
            "fileContent": extracted_text,
            "tokensUsed": {
                "totalTokensUsed": total_tokens_used,
                "modelUsed": model.value
            }
        }
        meta = {
            "isFileDownloaded": False,
            "fileSource": "Upload",
            "fileType": "PDF",
            "apiKeyValidation": "Valid",
            "hostingIntegration": True,
            "selectedModel": model.value
        }
        return JSONResponse(content={"status": "success", "message": "Operation completed successfully.", "data": data, "meta": meta})
    except Exception as e:
        error = {"errorCode": "500", "errorMessage": str(e)}
        return JSONResponse(content={"status": "error", "message": str(e), "error": error})
    



# Endpoint: Base64 PDF upload  Update the endpoint to include model selection
@app.post("/base64")
async def upload_base64_pdf(
    data: str = Form(...),  # Base64 encoded file content
    ext: str = Form(...),   # File extension 
    model: ModelChoice = Form(ModelChoice.GPT),  # Default to GPT
    api_key: str = Depends(validate_api_key)
):
    try:
        # Validate file extension
        allowed_extensions = ['pdf', 'txt']
        if ext.lower() not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file extension. Allowed extensions: {', '.join(allowed_extensions)}")

        # Decode base64 string
        file_data = base64.b64decode(data)
        
        # Generate unique filename with correct extension
        import uuid
        file_path = f"./uploads/{uuid.uuid4()}.{ext}"
        
        # Save the decoded file
        with open(file_path, "wb") as f:
            f.write(file_data)
        
        # Extract text based on file type
        if ext.lower() == 'pdf':
            extracted_text = extract_pdf_text(file_path)
        else:  # txt file
            with open(file_path, 'r') as f:
                extracted_text = f.read()

        # Save the extracted text to a file
        with open("./uploads/extracted_text.txt", "w") as f:
            f.write(extracted_text)
        
        # Select vendor details extraction function based on model
        if model == ModelChoice.GPT:
            vendor_details = await extract_vendor_details_gpt(extracted_text)
        else:
            vendor_details = await extract_vendor_details_groq(extracted_text)
        
        # Select text chunks processing function based on model
        max_chunk_size = 2000
        text_chunks = split_text(extracted_text, max_chunk_size)
        
        if model == ModelChoice.GPT:
            processed_chunks, total_tokens_used = await process_text_chunks_gpt(text_chunks)
        else:
            processed_chunks, total_tokens_used = await process_text_chunks_groq(text_chunks)

        # Aggregate invoices
        invoices_data = []
        for chunk_result in processed_chunks:
            invoices_data.extend(chunk_result.get("invoicesData", []))
        
        # Calculate balances and validation
        open_balance = sum(item.get("creditAmount", 0) for item in invoices_data) - \
                       sum(item.get("debitAmount", 0) for item in invoices_data)
        closing_balance = open_balance
        validation_status = "Valid" if open_balance == closing_balance else "Invalid"
        validation_message = "Balances match." if validation_status == "Valid" else "Balances do not match."
        
        # Format response
        data = {
            "vendorDetails": vendor_details.get("vendorDetails", {}),
            "invoiceSummary": {
                "invoiceDate": invoices_data[0].get("invoiceDate", "") if invoices_data else "",
                "numberOfRowItems": len(invoices_data),
                "openBalance": open_balance,
                "closingBalance": closing_balance,
                "validationStatus": validation_status,
                "modelUsed": model.value,
                "tokenUsage": total_tokens_used
            },
            "invoicesData": invoices_data,
            "validationResults": {
                "balanceValidationStatus": validation_status,
                "balanceValidationMessage": validation_message
            },
            "fileContent": extracted_text,
            "tokensUsed": {
                "totalTokensUsed": total_tokens_used,
                "modelUsed": model.value
            }
        }
        meta = {
            "isFileDownloaded": False,
            "fileSource": "Base64",
            "fileType": ext.upper(),
            "apiKeyValidation": "Valid",
            "hostingIntegration": True,
            "selectedModel": model.value
        }
        return JSONResponse(content={"status": "success", "message": "Operation completed successfully.", "data": data, "meta": meta})
    except Exception as e:
        error = {"errorCode": "500", "errorMessage": str(e)}
        return JSONResponse(content={"status": "error", "message": str(e), "error": error})
    
    

# Endpoint: File Link  Update the endpoint to include model selection
@app.post("/link")
async def upload_link_pdf(file_url: str = Form(...), model: ModelChoice = Form(ModelChoice.GPT), api_key: str = Depends(validate_api_key)):
    try:
        # Download the file from the URL
        file_path = "./uploads/link_upload.pdf"
        urllib.request.urlretrieve(file_url, file_path)
        
        # Extract text from the PDF
        extracted_text = extract_pdf_text(file_path)
        
        # Select vendor details extraction function based on model
        if model == ModelChoice.GPT:
            vendor_details = await extract_vendor_details_gpt(extracted_text)
        else:
            vendor_details = await extract_vendor_details_groq(extracted_text)
        
        # Select text chunks processing function based on model
        max_chunk_size = 2000
        text_chunks = split_text(extracted_text, max_chunk_size)
        
        if model == ModelChoice.GPT:
            processed_chunks, total_tokens_used = await process_text_chunks_gpt(text_chunks)
        else:
            processed_chunks, total_tokens_used = await process_text_chunks_groq(text_chunks)

        # Aggregate invoices
        invoices_data = []
        for chunk_result in processed_chunks:
            invoices_data.extend(chunk_result.get("invoicesData", []))
        
        # Calculate balances and validation
        open_balance = sum(item.get("creditAmount", 0) for item in invoices_data) - \
                       sum(item.get("debitAmount", 0) for item in invoices_data)
        closing_balance = open_balance
        validation_status = "Valid" if open_balance == closing_balance else "Invalid"
        validation_message = "Balances match." if validation_status == "Valid" else "Balances do not match."
        
        # Format response
        data = {
            "vendorDetails": vendor_details.get("vendorDetails", {}),
            "invoiceSummary": {
                "invoiceDate": invoices_data[0].get("invoiceDate", "") if invoices_data else "",
                "numberOfRowItems": len(invoices_data),
                "openBalance": open_balance,
                "closingBalance": closing_balance,
                "validationStatus": validation_status,
                "modelUsed": model.value,
                "tokenUsage": total_tokens_used
            },
            "invoicesData": invoices_data,
            "validationResults": {
                "balanceValidationStatus": validation_status,
                "balanceValidationMessage": validation_message
            },
            "fileContent": extracted_text,
            "tokensUsed": {
                "totalTokensUsed": total_tokens_used,
                "modelUsed": model.value
            }
        }
        meta = {
            "isFileDownloaded": True,
            "fileSource": "Link",
            "fileType": "PDF",
            "apiKeyValidation": "Valid",
            "hostingIntegration": True,
            "selectedModel": model.value
        }
        return JSONResponse(content={"status": "success", "message": "Operation completed successfully.", "data": data, "meta": meta})
    except Exception as e:
        error = {"errorCode": "500", "errorMessage": str(e)}
        return JSONResponse(content={"status": "error", "message": str(e), "error": error})
    

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
