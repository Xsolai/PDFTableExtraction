https://github.com/Xsolai/PDFTableExtraction.git

# PDF Data Extractor🤖

This FastAPI application extracts structured data from PDF files using OpenAI's GPT-4 model. The extracted data includes Reference number, Date, and Amount from the text within the PDF.

## Features

- Upload PDF files and extract structured data.
- Uses regex forinitial data extraction and validation.
- Utilizes OpenAI's GPT-4 model for enhanced data extraction.
- Caches results to improve performance andreduce API calls.
- Supports CORS for cross-origin requests.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- PyMuPDF (fitz)
- Pandas
- OpenAI Python client
- python-dotenv

## Installation

    1. Clone the repository:

```
git clone https://github.com/Xsolai/PDFTableExtraction.git
```

```
cd PDFTableExtraction
```

    2. Create and activate a virtual environment:

```
python -m venv venv
```

```
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

    3. Install the required packages:

```
pip install -r requirements.txt
```

    4. Create a`.env `filei n the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

    1. Run the FastAPIapplication:

```
uvicorn app:app --reload
```

    2. Open your browser and navigate to`http://127.0.0.1:8000/docs` to access the interactive API documentation.

    3. Use the`/upload-pdf/` endpoint to upload a PDF file and extract data.
