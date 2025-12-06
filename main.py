import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading local Qwen model...")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

print("Model loaded successfully.")

app = FastAPI(title="Infrakonnect Search API (Local Qwen)")


# -------------------------- MODELS --------------------------

class SearchRequest(BaseModel):
    query: str

class SearchMatch(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

class SearchResponse(BaseModel):
    keywords: List[str]
    matches: List[SearchMatch]


# -------------------------- KEYWORD EXTRACTOR --------------------------

def extract_keywords_local(query: str) -> List[str]:
    """
    Uses the local Qwen model to extract 3–7 keywords.
    Returns a JSON array of strings.
    """

    # UPDATED STRONGER PROMPT 💪
    prompt = f"""
You are a keyword extraction engine.

Your job:
- Extract 3 to 7 short keywords or phrases.
- Do NOT return the full sentence.
- Return ONLY a JSON array of strings.
- No explanation. No extra text.

Example output:
["roads", "infrastructure", "India", "2023", "development"]

Now extract keywords for this:
"{query}"
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.1
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract JSON array
    import json
    try:
        start = full_text.find("[")
        end = full_text.rfind("]") + 1
        json_str = full_text[start:end]
        return json.loads(json_str)
    except:
        return [query]


# -------------------------- FAKE DB SEARCH --------------------------

def fake_db_search(keywords: List[str]):
    joined = ", ".join(keywords)
    return [
        SearchMatch(
            title="Infrastructure Example Result",
            url="/services/example",
            snippet=f"Matches found for: {joined}"
        )
    ]


# -------------------------- MAIN ENDPOINT --------------------------

@app.post("/search", response_model=SearchResponse)
def search_api(body: SearchRequest):
    query = body.query
    keywords = extract_keywords_local(query)
    matches = fake_db_search(keywords)

    return SearchResponse(
        keywords=keywords,
        matches=matches
    )
