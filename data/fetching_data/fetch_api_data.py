import os
import csv
import json
import requests

from dotenv import load_dotenv
load_dotenv()

# Get API key from .env file
api_key = os.getenv("API_KEY")
 
url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={api_key}"
 
output_dir = "data/dataset"
os.makedirs(output_dir, exist_ok=True)  
csv_file_name = os.path.join(output_dir, "EnergyData.csv")

# Initialize list to store all records
all_records = []
 
def fetch_data():
    """
    Fetches hourly energy generation data from the EIA API in paginated requests
    and stores it in the `all_records` list.
    """
    offset = 0
    length = 5000
    while True:
        # Update the offset for the request
        headers = {
            "X-Params": json.dumps({
                "frequency": "hourly",
                "data": [
                    "value"
                ],
                "facets": {
                    "respondent": [
                        "NY"
                    ],
                    "fueltype": [
                        "WAT"
                    ]
                },
                "start": "2022-01-01T00",
                "end": "2024-12-01T00",
                "sort": [
                    {
                        "column": "period",
                        "direction": "desc"
                    }
                ],
                "offset": offset,
                "length": length
            })
        }

        # Make GET request
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Extract records from the response
            records = data.get("response", {}).get("data", [])
            
            if not records:
                break  # Stop if no more records are returned

            # Add records to the list
            all_records.extend(records)

            # If we have fetched the required rows, stop fetching
            if len(all_records) >= 60000:
                break

            # Update the offset for the next batch
            offset += length

        else:
            print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
            print(f"Error message: {response.text}")
            break

# Fetch all data
fetch_data()

# Check if we have enough records
if all_records:
    # Dynamically determine all columns from the first record
    all_columns = list(all_records[0].keys())

    # Write all data to CSV
    with open(csv_file_name, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_columns)

        # Write header
        writer.writeheader()

        # Write all records
        for record in all_records:
            writer.writerow(record)

    print(f"All data saved successfully to {csv_file_name}")
    print(f"Total rows saved: {len(all_records)}")
else:
    print("Error: No data retrieved.")
