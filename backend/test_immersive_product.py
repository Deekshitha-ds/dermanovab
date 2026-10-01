import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# ------------------------------------------------------------
# STEP 1: Get Google Shopping results
# ------------------------------------------------------------

shopping_params = {
    "engine": "google_shopping",
    "q": "The Ordinary Niacinamide 10% Serum face serum",
    "api_key": SERPAPI_KEY,
    "google_domain": "google.co.in",
    "gl": "in",
    "hl": "en",
    "num": "10",
}

shopping_response = requests.get(
    "https://serpapi.com/search.json",
    params=shopping_params,
    timeout=30,
)

shopping_response.raise_for_status()

shopping_data = shopping_response.json()

shopping_results = shopping_data.get(
    "shopping_results",
    []
)

if not shopping_results:
    print("No shopping results found.")
    raise SystemExit


# ------------------------------------------------------------
# STEP 2: Get the first product token
# ------------------------------------------------------------

first_product = shopping_results[0]

token = first_product.get(
    "immersive_product_page_token"
)

if not token:
    print("No immersive product token found.")
    raise SystemExit


print("\nProduct:")
print(first_product.get("title"))

print("\nStore:")
print(first_product.get("source"))

print("\nPrice:")
print(first_product.get("price"))

print("\nToken found successfully.")
print("=" * 80)


# ------------------------------------------------------------
# STEP 3: Request immersive product details
# ------------------------------------------------------------

product_params = {
    "engine": "google_immersive_product",
    "page_token": token,
    "api_key": SERPAPI_KEY,
    "more_stores": "true",
}

product_response = requests.get(
    "https://serpapi.com/search.json",
    params=product_params,
    timeout=30,
)

print("\nHTTP STATUS:", product_response.status_code)

product_response.raise_for_status()

product_data = product_response.json()


# ------------------------------------------------------------
# STEP 4: Display result
# ------------------------------------------------------------

print("\nIMMERSIVE PRODUCT RESULT")
print("=" * 80)

print(
    json.dumps(
        product_data,
        indent=2,
        ensure_ascii=False
    )
)



print("\n\nUSEFUL LINK / STORE INFORMATION")
print("=" * 80)

def find_links(obj, path="root"):
    if isinstance(obj, dict):
        for key, value in obj.items():

            key_lower = str(key).lower()

            if (
                "link" in key_lower
                or "url" in key_lower
                or "store" in key_lower
                or "merchant" in key_lower
                or "seller" in key_lower
                or "price" in key_lower
            ):
                print(f"\nPATH: {path}")
                print(f"FIELD: {key}")
                print(f"VALUE: {value}")

            find_links(
                value,
                f"{path}.{key}"
            )

    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            find_links(
                item,
                f"{path}[{index}]"
            )


find_links(product_data)