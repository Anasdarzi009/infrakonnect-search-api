import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

print("Loading local Qwen model...")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

print("Model loaded successfully.")

app = FastAPI(title="Infrakonnect APIs (Local Qwen)")

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


class BlogRequest(BaseModel):
    topic: str
    word_count: Optional[int] = 800  # default length


class BlogResponse(BaseModel):
    title: str
    slug: str
    summary: str
    content: str
    keywords: List[str]


# -------------------------- KEYWORD EXTRACTOR (SEARCH) --------------------------

def extract_keywords_local(query: str) -> List[str]:
    """
    Uses the local Qwen model to extract 3–7 keywords.
    Returns a JSON array of strings.
    """

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
    try:
        start = full_text.find("[")
        end = full_text.rfind("]") + 1
        json_str = full_text[start:end]
        return json.loads(json_str)
    except Exception:
        return [query]


# -------------------------- FAKE DB SEARCH --------------------------

def fake_db_search(keywords: List[str]) -> List[SearchMatch]:
    joined = ", ".join(keywords)
    return [
        SearchMatch(
            title="Infrastructure Example Result",
            url="/services/example",
            snippet=f"Matches found for: {joined}"
        )
    ]


# -------------------------- BLOG GENERATOR --------------------------

def generate_blog_local(topic: str, word_count: int = 800) -> BlogResponse:
    """
    Uses the local Qwen model to generate a structured blog:
    - title
    - slug
    - summary
    - content
    - keywords
    """

    prompt = f"""
You are an AI content writer for an infrastructure company website.

Write a detailed blog about the topic below.

Topic: "{topic}"
Target word count (approx): {word_count}

Return the result STRICTLY as a JSON object with this exact structure:

{{
  "title": "...",
  "slug": "...",
  "summary": "...",
  "content": "...",
  "keywords": ["...", "...", "..."]
}}

Rules:
- "slug" should be URL friendly: lowercase, hyphens, no special characters.
- "summary" should be 2–4 lines, a short overview.
- "content" should be multi-paragraph blog text (no JSON, no markdown).
- "keywords" should have 3–7 short phrases.
- Do NOT add any text before or after the JSON. Only return the JSON object.
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.7,
            top_p=0.9
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to parse JSON object from model output
    try:
        start = full_text.find("{")
        end = full_text.rfind("}") + 1
        json_str = full_text[start:end]
        data = json.loads(json_str)
    except Exception:
        # Fallback: if parsing fails, build a simple blog
        data = {
            "title": topic,
            "slug": topic.lower().replace(" ", "-"),
            "summary": f"A blog about: {topic}",
            "content": full_text,
            "keywords": [topic]
        }

    # Ensure all fields exist
    title = str(data.get("title", topic))
    slug = str(data.get("slug", topic.lower().replace(" ", "-")))
    summary = str(data.get("summary", f"A blog about: {topic}"))
    content = str(data.get("content", full_text))
    keywords = data.get("keywords", [topic])

    if not isinstance(keywords, list):
        keywords = [str(keywords)]

    keywords = [str(k) for k in keywords]

    return BlogResponse(
        title=title,
        slug=slug,
        summary=summary,
        content=content,
        keywords=keywords
    )


# -------------------------- ENDPOINTS --------------------------

@app.post("/search", response_model=SearchResponse)
def search_api(body: SearchRequest):
    query = body.query
    keywords = extract_keywords_local(query)
    matches = fake_db_search(keywords)

    return SearchResponse(
        keywords=keywords,
        matches=matches
    )


@app.post("/blog/generate", response_model=BlogResponse)
def blog_generate_api(body: BlogRequest):
    """
    Generate a blog for a given topic using the local Qwen model.
    """
    return generate_blog_local(body.topic, body.word_count)
