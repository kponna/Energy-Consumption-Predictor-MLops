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
# collection = db["energy_generation"]

# # Function to fetch data from API with pagination support
# def fetch_data(api_key, offset=0, length=5000):
#     url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={api_key}"
    
#     headers = {
#         "X-Params": json.dumps({
#             "frequency": "hourly",
#             "data": ["value"],  # Data of interest
#             "facets": {
#                 "respondent": ["NY"],
#                 "fueltype": ["WAT"]
#             },
#             "start": "2019-01-01T00",
#             "end": "2024-11-01T00",
#             "sort": [
#                 {"column": "period", "direction": "desc"}
#             ],
#             "offset": offset,
#             "length": length
#         })
#     }

#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         data = response.json().get("response", {}).get("data", [])
#         # Filter only `period` and `value` fields
#         filtered_data = [{"period": item["period"], "value": item["value"]} for item in data]
#         return filtered_data
#     else:
#         print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
#         print(f"Error message: {response.text}")
#         return []

# # Total rows required
# total_rows = 52000
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
#     for record in fetched_data:
#         collection.update_one(
#             {"period": record["period"], "value": record["value"]},  # Match on unique fields
#             {"$set": record},  # Update or insert the record
#             upsert=True  # Insert if it doesn't exist
#         )
#     print("Data inserted/updated in MongoDB successfully.")


#     # Fetch data from MongoDB and convert to pandas DataFrame
#     cursor = collection.find({}, {"_id": 0})  # Exclude MongoDB's default _id field
#     df = pd.DataFrame(list(cursor))

#     # Display DataFrame
#     print(df.head())
# else:
#     print("No data fetched from the API.")

#---------------------------------------------------------
# Delete the documents

from pymongo import MongoClient
import os
# MongoDB connection
mongo_uri = os.getenv("MONGO_URI") # Replace with your actual MongoDB URI
client = MongoClient(mongo_uri)

# Select the database and collection
db = client["eia_data"]
collection = db["energy_generation"]

# Delete all documents
result = collection.delete_many({})  # Pass an empty filter to delete all documents
print(f"{result.deleted_count} documents deleted.")
