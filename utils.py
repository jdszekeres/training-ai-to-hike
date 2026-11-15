import time
import requests
import os

def download_file(url, local_filename):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses

    # Create the local file path
    local_path = os.path.join(os.getcwd(), "data", local_filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Write the content to a local file
    with open(local_path, 'wb+') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path

def overpass_query(query, count = 0):
    """
    Sends a query to the Overpass API and returns the response.
    
    Parameters:
    query: The Overpass QL query string.
    
    Returns:
    The JSON response from the Overpass API.
    """
    url = "http://overpass-api.de/api/interpreter"
    response = requests.post(url, data={'data': query}, headers={'Content-Type': 'application/x-www-form-urlencoded', 'User-Agent': 'RLHiking/1.0 (jdszekeres@gmail.com)'})
    if response.status_code == 502:
        if count >= 5:
            raise Exception("Overpass API is down or not responding after multiple attempts.")
        time.sleep(10 ** count)  # Exponential backoff
        return overpass_query(query, count + 1)  # Retry the query
        
    response.raise_for_status()  # Raise an error for bad responses
    return response.json()

def node_id_to_pixel(node_id):
    #Use overpass_query to get the node's coordinates
    query = f"""
    [out:json];
    node({node_id});
    out body;
    """
    response = overpass_query(query)



