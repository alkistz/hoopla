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
