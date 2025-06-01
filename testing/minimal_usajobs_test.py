import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

USAJOBS_API_KEY = os.getenv("USAJOBS_API_KEY")
USAJOBS_USER_AGENT = os.getenv("USAJOBS_USER_AGENT")

print(f"--- Minimal USAJOBS API Test ---")
print(f"Loaded API Key: {USAJOBS_API_KEY[:5]}..." if USAJOBS_API_KEY else "API Key NOT FOUND")
print(f"Loaded User Agent: {USAJOBS_USER_AGENT}" if USAJOBS_USER_AGENT else "User Agent NOT FOUND")

if not USAJOBS_API_KEY or not USAJOBS_USER_AGENT:
    print("Error: API Key or User Agent not found in .env file. Please check.")
else:
    url = "https://data.usajobs.gov/api/Search" # Matches your cURL URL

    # --- RECTIFICATION ---
    # To match your cURL command which had no query parameters,
    # we will pass an empty dictionary for params.
    # If you intend to search with parameters, you can re-add them here,
    # but ensure they are correct according to the API documentation.
    params = {}
    # Original params that were different from your cURL:
    # params = {
    # "JobCategoryCode": "2210"
    # }

    headers = {
        'User-Agent': USAJOBS_USER_AGENT,
        'Authorization-Key': USAJOBS_API_KEY,
        'Host': 'data.usajobs.gov' # As per USAJOBS documentation
    }

    print(f"\nRequesting URL: {url}")
    print(f"With Params: {params}") # Will show {} if params is empty
    print(f"With Headers (excluding key for print): User-Agent: {headers['User-Agent']}, Host: {headers['Host']}")

    try:
        # If params is an empty dictionary or None, no query string will be appended
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        print(f"\nStatus Code: {response.status_code}")
        print("Response Headers:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        print("\nResponse Content (first 500 chars):")
        print(response.text[:500])

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        print("\nJSON Response Parsed Successfully:")
        # print(json.dumps(data, indent=2)) # Pretty print the JSON

        # Check if 'SearchResult' and 'SearchResultCount' exist before accessing
        search_result = data.get('SearchResult', {})
        search_result_count = search_result.get('SearchResultCount', 0)
        search_result_count_all = search_result.get('SearchResultCountAll', 0)


        if search_result_count > 0:
            print(f"\nSuccessfully fetched {search_result_count} job(s) (SearchResultCount) matching the query (if any)!")
        elif search_result_count_all > 0 : # Some APIs might return SearchResultCountAll even if SearchResultCount is 0 for broad queries
             print(f"\nRequest successful. Total jobs available matching broader criteria: {search_result_count_all} (SearchResultCountAll). Current query returned 0 specific results.")
        else:
            print("\nRequest successful, but no jobs found for this query or the endpoint didn't return job counts in the expected way for a parameter-less query.")
            if 'error' in str(data).lower() or 'message' in str(data).lower():
                 print(f"The API might have returned a message: {data}")


    except requests.exceptions.HTTPError as http_err:
        print(f"\nHTTP error occurred: {http_err}")
        print(f"Response content: {response.text}") # Print full response text on HTTP error
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")
    except json.JSONDecodeError:
        print("\nError decoding JSON from the response. The API might not have returned valid JSON.")
        print(f"Response content that caused JSON error: {response.text}")