from pymongo import MongoClient

# Connect to MongoDB Atlas
def get_data():
    # REPLACE WITH YOUR MONOGDB CONNECTION STRING
    client = MongoClient("<YOUR_CONNECTION_STRING>")
    db = client.agriculture
    collection = db.soil_data

    # Assuming there is a 'timestamp' field that records the document's creation time
    latest_document = collection.find().sort("created_at", -1).limit(1)
    print(latest_document)
    # Print the latest document
    for doc in latest_document:
        return doc