import json
import os

from dotenv import load_dotenv
from google import genai


def create_gemini_client():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("There is not API key")

    return genai.Client(api_key=api_key)


def improve_query(query: str, method: str):
    try:
        client = create_gemini_client()
        prompt = query
        match method:
            case "spell":
                prompt = enchance_prompt(query)
            case "rewrite":
                prompt = rewrite_prompt(query)
            case "expand":
                prompt = expand_prompt(query)

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
        )
        print(f"Enhanced query ({method}): '{query}' -> '{response.text}'\n")

        return response.text
    except ValueError:
        print("No API key")


def score_movie(query: str, doc):
    try:
        client = create_gemini_client()
        prompt = rerank_prompt(query, doc)

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
        )

        return response.text
    except ValueError:
        print("No API key")


def batch_rank_movies(query: str, doc_list_str: list) -> list | None:
    try:
        client = create_gemini_client()
        prompt = rerank_batch_prompt(query, doc_list_str)

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
        )

        if not response.text:
            return []

        return json.loads(response.text)
    except ValueError:
        print("No API key")


def evaluate_results(query: str, formatted_results: list):
    client = create_gemini_client()
    prompt = llm_evaluation_prompt(query, formatted_results)

    response = client.models.generate_content(
        model="gemini-2.0-flash-001", contents=prompt
    )

    if not response.text:
        return []

    return json.loads(response.text)


def rag_response(query: str, docs):
    client = create_gemini_client()
    prompt = rag_prompt(query, docs)

    response = client.models.generate_content(
        model="gemini-2.0-flash-001", contents=prompt
    )

    if not response.text:
        return ""
    return response.text


def enchance_prompt(query: str) -> str:
    return f"""
Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""


def rewrite_prompt(query: str) -> str:
    return f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""


def expand_prompt(query: str) -> str:
    return f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""


def rerank_prompt(query: str, doc) -> str:
    return f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""


def rerank_batch_prompt(query: str, doc_list_str: list) -> str:
    return f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""


def llm_evaluation_prompt(query: str, formatted_results: list):
    return f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""


def rag_prompt(query: str, docs):
    return f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""
