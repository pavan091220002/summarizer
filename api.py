from fastapi import FastAPI
from utils import process_company

app = FastAPI()

@app.get("/analyze/{company}")
async def analyze_company(company: str):
    """API endpoint to analyze a company."""
    result = process_company(company)
    return result