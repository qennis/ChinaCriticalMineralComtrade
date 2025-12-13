import requests

# The exact URL from your failure log
url = "https://comtradeapi.un.org/data/v1/get/C/M/HS"
params = {
    "reporterCode": "156",
    "partnerCode": "842",
    "flowCode": "X",
    # A chunk of 10 codes (reduced from 20 to test limits)
    "cmdCode": "250410,250490,260200,260300,260400,260500,280461,280469,280520,280530",
    "period": "202201",
    "subscription-key": "c0b717eaf75e4a5682bcd61cab5b300e",  # Your key from the logs
}

print(f"Testing Request to: {url}")
try:
    r = requests.get(url, params=params, timeout=30)
    print(f"Status Code: {r.status_code}")
    print("Response Body:")
    print(r.text[:1000])  # Print first 1000 chars of response
except Exception as e:
    print(f"Error: {e}")
