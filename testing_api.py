import requests
import json
import csv
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key and MongoDB connection string from .env file
api_key = os.getenv("API_KEY")
mongo_uri = os.getenv("MONGO_URI")  

# API URL
url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={api_key}"

# Define the CSV file path
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
csv_file_name = os.path.join(output_dir, "energy_generated.csv")

# Initialize list to store all records
all_records = []

# Function to fetch data in paginated requests
def fetch_data():
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
                "start": "2023-11-01T00",
                "end": "2024-11-01T00",
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
            if len(all_records) >= 24000:
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
    # Specify the columns to extract
    selected_columns = ["period", "value"]
    
    # Write selected data to CSV
    with open(csv_file_name, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=selected_columns)
        
        # Write header
        writer.writeheader()
        
        # Write filtered records
        for record in all_records:
            filtered_record = {key: record[key] for key in selected_columns if key in record}
            writer.writerow(filtered_record)
    
    print(f"Filtered data saved successfully to {csv_file_name}")
    print(f"Total rows saved: {len(all_records)}")
else:
    print("Error: No data retrieved.")  
# ----------------------------------------------
# import os
# import json
# import requests
# import pandas as pd
# from dotenv import load_dotenv
# from pymongo import MongoClient

# # Load environment variables
# load_dotenv()

# # Get API key and MongoDB connection string from .env file
# api_key = os.getenv("API_KEY")
# mongo_uri = os.getenv("MONGO_URI")

# # MongoDB connection
# client = MongoClient(mongo_uri)
# db = client["eia_data"]
# collection = db["region_consumption"]

# # Function to fetch data from API with pagination support
# def fetch_data(api_key, start_offset=0, length=5000):
#     url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?api_key={api_key}"
    
#     headers = {
#         "X-Params": json.dumps({
#             "frequency": "hourly",
#             "data": ["value"],
#             "facets": {
#                 "type": ["CO2.DD"],
#                 "respondent": [
#                     "AVA", "CAL", "CISO", "IID", "LDWP", 
#                     "PACE", "PACW", "PGE", "SCL", "SRP", "TPWR"
#                 ]
#             },
#             "start": "2019-01-01T00",
#             "end": "2024-11-29T00",
#             "sort": [{"column": "period", "direction": "desc"}],
#             "offset": start_offset,
#             "length": length
#         })
#     }

#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         return response.json().get("response", {}).get("data", [])
#     else:
#         print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
#         print(f"Error message: {response.text}")
#         return []

# # Total rows required
# total_rows = 6603
# batch_size = 5000  # Max rows per request
# offset = 0
# fetched_data = []

# # Fetch data in batches if necessary
# while offset < total_rows:
#     print(f"Fetching data from offset {offset}...")
#     data = fetch_data(api_key, offset, batch_size)
#     if not data:
#         break
#     fetched_data.extend(data)
#     offset += batch_size

# # Check if we fetched all the required data
# if len(fetched_data) == total_rows:
#     print(f"Successfully fetched {len(fetched_data)} rows.")
# else:
#     print(f"Fetched {len(fetched_data)} rows, which is less than expected.")

# # Insert data into MongoDB
# if fetched_data:
#     collection.insert_many(fetched_data)
#     print("Data inserted into MongoDB successfully.")

#     # Fetch data from MongoDB and convert to pandas DataFrame
#     cursor = collection.find({}, {"_id": 0})  # Exclude MongoDB's default _id field
#     df = pd.DataFrame(list(cursor))

#     # Display DataFrame
#     print(df.head())
# else:
#     print("No data fetched from the API.")
