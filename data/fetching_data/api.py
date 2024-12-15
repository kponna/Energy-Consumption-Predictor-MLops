import requests
import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Get API key and MongoDB connection string from .env file
api_key = os.getenv("API_KEY")
mongo_uri = os.getenv("PYMONGO_URI")  

# API URL
url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={api_key}"

# MongoDB Setup
try:
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client['energy_db']  # Database name
    collection = db['energy_data']  # Collection name
    print("Connected to MongoDB successfully!")

except Exception as e:
    raise Exception(f"Failed to connect to MongoDB: {e}")

# Function to fetch and store data in MongoDB
def fetch_and_store_data():
    """
    Fetch data in paginated requests from the API and store it into MongoDB.
    """
    offset = 0
    length = 5000
    total_inserted = 0

    while True:
        # Define headers for paginated request
        headers = {
            "X-Params": json.dumps({
                "frequency": "hourly",
                "data": ["value"],
                "facets": {
                    "respondent": ["NY"],
                    "fueltype": ["WAT"]
                },
                "start": "2022-01-01T00",
                "end": "2024-12-01T00",
                "sort": [{"column": "period", "direction": "desc"}],
                "offset": offset,
                "length": length
            })
        }

        # Make GET request to the API
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            records = data.get("response", {}).get("data", [])
            
            # Break the loop if no more records are returned
            if not records:
                print("No more records to fetch.")
                break

            # Insert records into MongoDB
            try:
                collection.insert_many(records)
                total_inserted += len(records)
                print(f"{len(records)} records inserted. Total so far: {total_inserted}")
            except Exception as e:
                print(f"Failed to insert records into MongoDB: {e}")
                break

            # Stop fetching if total rows reach or exceed 60,000
            if total_inserted >= 60000:
                print("Reached the required limit of 60,000 records.")
                break

            # Update offset for the next request
            offset += length

        else:
            # Print error message for failed API call
            print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
            print(f"Error message: {response.text}")
            break

# Run the function to fetch and store data
if __name__ == "__main__":
    try:
        fetch_and_store_data()
        print("Data successfully stored in MongoDB!")
    except Exception as e:
        print(f"An error occurred: {e}")

# import requests
# import pandas as pd
# import os
# import json
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # API Key and MongoDB URI from .env file
# api_key = os.getenv("API_KEY")
# mongo_uri = os.getenv("MONGO_URI")

# # API Endpoint URL
# BASE_URL = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={api_key}"

# # Define output CSV file path
# OUTPUT_DIR = "data/dataset"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# CSV_FILE_PATH = os.path.join(OUTPUT_DIR, "energy_generated.csv")

# # Function to fetch paginated data
# def fetch_energy_data(api_url, limit_rows=35000, batch_size=5000):
#     """
#     Fetch paginated energy data from the API.

#     :param api_url: Base API endpoint URL.
#     :param limit_rows: Maximum number of rows to fetch.
#     :param batch_size: Number of rows to fetch per request.
#     :return: DataFrame containing fetched energy data.
#     """
#     offset = 0
#     all_records = []  # List to store all fetched data

#     # Start fetching data in paginated requests
#     while len(all_records) < limit_rows:
#         print(f"Fetching records {offset + 1} to {offset + batch_size}...")

#         # Create request headers and parameters
#         headers = {
#             "X-Params": json.dumps({
#                 "frequency": "hourly",
#                 "data": ["value"],
#                 "facets": {
#                     "respondent": ["NY"],
#                     "fueltype": ["WAT"]
#                 },
#                 "start": "2022-01-01T00",
#                 "end": "2024-12-01T00",
#                 "sort": [{"column": "period", "direction": "desc"}],
#                 "offset": offset,
#                 "length": batch_size
#             })
#         }

#         # API Request
#         response = requests.get(api_url, headers=headers)
        
#         # Check for success
#         if response.status_code == 200:
#             data = response.json()
#             # Extract records from the response
#             records = data.get("response", {}).get("data", [])
            
#             if not records:
#                 print("No more records returned. Stopping fetch.")
#                 break  # Stop if no records are returned
            
#             all_records.extend(records)
#             offset += batch_size  # Update offset for next request
#         else:
#             raise Exception(f"API Error: {response.status_code} - {response.text}")

#     print(f"Total records fetched: {len(all_records)}")
#     return pd.DataFrame(all_records)  # Convert to DataFrame

# # Main execution block
# if __name__ == "__main__":
#     try:
#         print("Starting data fetch from eia api...")
        
#         # Fetch data
#         energy_data = fetch_energy_data(BASE_URL)

#         if not energy_data.empty:
#             # Select required columns
#             selected_columns = ["period", "value"]
#             filtered_data = energy_data[selected_columns]

#             # Save data to CSV
#             filtered_data.to_csv(CSV_FILE_PATH, index=False)
#             print(f"Filtered data saved successfully to {CSV_FILE_PATH}")
#             print(f"Total rows saved: {len(filtered_data)}")
#         else:
#             print("No data retrieved from the API.")
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
