# src/api.py
import os
import logging
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

from src.config import BASE_URL, PREF_CODE, TIMEOUT

logger = logging.getLogger(__name__)

def get_api_key() -> str:
    """
    Loads and validates the API key.
    """
    load_dotenv()
    key = os.getenv('MLIT_API_KEY')
    if not key:
        raise ValueError("Missing API Key. Please check your .env file.")
    return key

def fetch_year_data(api_key: str, year: int) -> List[Dict[str, Any]]:
    """
    Fetches a single year of data from the MLIT API.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }

    params = {
        "area": PREF_CODE,
        "year": year,
        "language": "en"
    }
    
    try:
        response = requests.get(BASE_URL, headers=headers, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        # The API returns a dict with a "data" key
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data for year {year}: {e}")
        return []