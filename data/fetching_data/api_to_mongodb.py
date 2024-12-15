import os
import json
import sys
sys.path.append('/home/karthikponna/kittu/Energy generation prediction project/Energy-Generation-Predictor-MLops')
import requests
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

from energygeneration.logging.logger import logging
from energygeneration.exception_handling.exception import EnergyGenerationException
 
class EnergyDataExtract():
    def __init__(self):
        try:
            # Load environment variables
            load_dotenv()
            self.MONGO_URI = os.getenv("PYMONGO_URI")
            self.API_KEY = os.getenv("API_KEY")
            
            # Check for missing environment variables
            if not self.MONGO_URI or not self.API_KEY:
                logging.error("Missing MongoDB URI or API Key in environment variables.")
                raise EnergyGenerationException("Environment variables for MONGO_URI or API_KEY not found.", sys)

            # MongoDB Client setup
            self.client = MongoClient(self.MONGO_URI, tlsCAFile=certifi.where())
            self.db = self.client['eia_data']  # Database name
            self.collection = self.db['energy_generation']  # Collection name

        except Exception as e:
            raise EnergyGenerationException(e, sys)

    def fetch_and_store_in_mongodb(self):
        """
        Fetches hourly energy generation data from the EIA API and stores it in MongoDB.
        """
        try:
            # API URL
            API_URL = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={self.API_KEY}"

            # Find the latest entry in MongoDB
            latest_entry = self.collection.find_one(sort=[("period", -1)])
            if latest_entry:
                last_date = pd.to_datetime(latest_entry['period']).strftime('%Y-%m-%dT%H')
            else:
                last_date = '2022-01-01T00'  # Default start date

            logging.info(f"Starting data fetch from {last_date}...")

            # Pagination setup
            offset = 0
            length = 5000
            total_inserted = 0

            while True:
                # Define API headers with parameters
                headers = {
                    "X-Params": json.dumps({
                        "frequency": "hourly",
                        "data": ["value"],
                        "facets": {
                            "respondent": ["NY"],
                            "fueltype": ["WAT"]
                        },
                        "start": last_date,
                        "end": "2024-12-01T00",
                        "sort": [{"column": "period", "direction": "desc"}],
                        "offset": offset,
                        "length": length
                    })
                }

                # Make the GET request
                response = requests.get(API_URL, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    records = data.get("response", {}).get("data", [])

                    if not records:
                        logging.info("No more records to fetch.")
                        break

                    # Convert to DataFrame and filter new records
                    new_data_df = pd.DataFrame(records)
                    if not new_data_df.empty:
                        new_data_df['period'] = pd.to_datetime(new_data_df['period'])
                        if latest_entry:
                            new_data_df = new_data_df[new_data_df['period'] > pd.to_datetime(last_date)]

                        # Insert new records into MongoDB
                        if not new_data_df.empty:
                            data_to_insert = new_data_df.to_dict(orient='records')
                            result = self.collection.insert_many(data_to_insert)
                            total_inserted += len(result.inserted_ids)
                            logging.info(f"Inserted {len(result.inserted_ids)} records. Total: {total_inserted}.")
                        else:
                            logging.info("No new data to insert.")
                            break
                    else:
                        logging.warning("No records found in the API response.")
                        break

                    # Stop when total inserted records reach 30,000
                    if total_inserted >= 30000:
                        logging.info("Reached the limit of 30,000 records. Stopping fetch.")
                        break

                    # Update offset for pagination
                    offset += length
                else:
                    logging.error(f"API request failed. HTTP Status: {response.status_code}, Error: {response.text}")
                    raise EnergyGenerationException(f"API Request Failed: {response.text}", sys)

            logging.info("Data fetching and storage completed successfully!")

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise EnergyGenerationException(e, sys)

    def clear_collection(self):
        """ 
        This function allows you to drop all the records and start fresh.
        """
        try:
            # Drop all records in the collection
            result = self.collection.delete_many({})
            logging.info(f"Successfully deleted {result.deleted_count} records from the collection.")
        except Exception as e: 
            raise EnergyGenerationException(e, sys)

if __name__ == "__main__":

    energy_data_obj = EnergyDataExtract()
    energy_data_obj.fetch_and_store_in_mongodb() 
    
    # Uncomment to clear the collection before fetching data
    # energy_data_obj.clear_collection()
    