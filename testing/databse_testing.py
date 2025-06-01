import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure, ConfigurationError, ServerSelectionTimeoutError
from dotenv import load_dotenv

def test_mongodb_connection():
    """
    Tests the connection to MongoDB Atlas using a URI from environment variables.
    """
    # Load environment variables from a .env file (if you have one)
    load_dotenv()

    mongo_uri = os.environ.get("MONGO_ATLAS_URI")

    if not mongo_uri:
        print("ERROR: The MONGO_ATLAS_URI environment variable is not set.")
        print("Please ensure it's defined in your .env file or system environment.")
        return

    print(f"Attempting to connect to MongoDB Atlas using the provided URI...")
    # You can print the URI for debugging but be careful not to expose sensitive parts in logs
    # print(f"DEBUG: URI found: {mongo_uri[:mongo_uri.find('@') if '@' in mongo_uri else 30]}...")


    # Create a new client and connect to the server
    # It's good practice to set a server_api version for compatibility
    client = MongoClient(mongo_uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000) # 5 second timeout

    try:
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas!")
        print("Ping to your deployment was successful.")

        # Optionally, list databases as a further test (requires appropriate permissions)
        # print("\nListing databases (requires permissions):")
        # try:
        #     db_list = client.list_database_names()
        #     print(db_list)
        # except Exception as e_db_list:
        #     print(f"Could not list databases: {e_db_list}")

    except ConfigurationError as e_config:
        print(f"MongoDB Configuration Error: {e_config}")
        print("This might be due to an invalid URI format or options.")
    except ServerSelectionTimeoutError as e_timeout:
        print(f"MongoDB Server Selection Timeout Error: {e_timeout}")
        print("Could not connect to the server within the timeout period.")
        print("Check your network connection, firewall settings, and if the MongoDB server/cluster is running and accessible.")
        print("Also, ensure your IP address is whitelisted in MongoDB Atlas if IP restrictions are enabled.")
    except ConnectionFailure as e_conn:
        print(f"MongoDB Connection Failure: {e_conn}")
        print("Failed to connect to the MongoDB server.")
        print("Check the URI, network, firewall, and server status.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Important: Close the connection
        if 'client' in locals() and client:
            client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    test_mongodb_connection()