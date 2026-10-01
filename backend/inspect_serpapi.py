import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

params = {
    "engine": "google_shopping",
    "q": "The Ordinary Niacinamide 10% Serum face serum",
    "api_key": SERPAPI_KEY,
    "google_domain": "google.co.in",
    "gl": "in",
    "hl": "en",
    "num": "10",
}

response = requests.get(
    "https://serpapi.com/search.json",
    params=params,
    timeout=30,
)

response.raise_for_status()

data = response.json()

results = data.get("shopping_results", [])

print("\nNUMBER OF RESULTS:", len(results))
print("=" * 80)

if results:
    first = results[0]

    print("\nFIRST RESULT - ALL FIELDS:")
    print("=" * 80)

    print(json.dumps(
        first,
        indent=2,
        ensure_ascii=False
    ))

else:
    print("No shopping results found.")