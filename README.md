## Invoice Data Extractor API 🔰

### Overview

This API is designed to process PDF invoices and extract essential data fields such as:

* **Invoice Number**
* **Date**
* **Amount**

It leverages OpenAI's GPT-3.5 model to assist with identifying the location of these fields in tabular data, supporting various date formats and handling preprocessing of amounts and invoice numbers.

### Features

* **PDF Parsing** : Extracts tables from PDF invoices using `tabula`.
* **Data Extraction** : Uses GPT-3.5 for identifying key fields from sample rows.
* **Data Preprocessing** : Handles different formats for dates, amounts, and invoice numbers to ensure clean output.

---

### Project Structure

The primary class, `InvoiceDataExtractor`, contains the following methods:

* **`preprocess_date`** : Standardizes various date formats to `YYYY-MM-DD`.
* **`preprocess_amount`** : Cleans and formats amounts as floating-point values.
* **`preprocess_invoice_number`** : Strips common prefixes to extract clean invoice numbers.
* **`analyze_sample_rows`** : Uses OpenAI GPT-3.5 to determine column mappings for key fields.
* **`extract_data_using_mappings`** : Extracts and preprocesses data from table rows based on column mappings.
* **`extract_from_pdf`** : Reads and processes tables from PDF files, aggregating results.

### Prerequisites

1. **Python 3.7+**
2. **Dependencies** :

* FastAPI
* Tabula
* Pandas
* OpenAI API client

To install these dependencies, run:

```
pip install -r requirements.txt
```

**Java** : `tabula-py` requires Java to be installed for PDF processing.

```
https://www.java.com/en/download/

```

After installation add this to system path and restart the pc!

### Setup

1. **OpenAI API Key** : Replace `api_key` in the `app.py` endpoint with your actual OpenAI API key.
2. **Run the FastAPI App** :

```
uvicorn app:app --relaod
```

### API Endpoints

#### 1. Extract Invoice Data

* **Endpoint** : `POST /extract-invoice-data`
* **Description** : Accepts a PDF file, extracts tabular invoice data, and returns the specified fields.
* **Request** : Multipart form data with an uploaded PDF file.
* **Response** :
* `200 OK`: JSON with extracted data.
* `400 Bad Request`: If no data is extracted.
* `500 Internal Server Error`: For any processing errors.
