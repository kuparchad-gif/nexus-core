
from fastapi import APIRouter, Query
from bs4 import BeautifulSoup
import requests

router = APIRouter()

@router.get("/browse")
def browse_url(url: str = Query(...)):
    try:
        headers = {"User-Agent": "EdenBridge-Agent"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        # Strip scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return {"content": text[:5000]}  # limit output
    except Exception as e:
        return {"error": str(e)}
