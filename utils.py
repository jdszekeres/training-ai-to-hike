import time
from mujoco import mujoco
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


def import_mujoco_xml_to_spec(spec, xml_string, frame_name="imported_frame"):
    """
    Imports a MuJoCo XML string into a Brax environment specification.
    
    Parameters:
    spec: The Brax environment specification object.
    xml_string: The MuJoCo XML content as a string.
    
    Returns:
    The updated Brax environment specification with the imported XML.
    """
    # Assuming spec has a method to import XML, this is a placeholder
    with mujoco.MjVfs() as vfs:
        snippet_xml = xml_string.encode('utf-8')
        vfs["snippet.xml"] = snippet_xml
        new_spec = mujoco.MjSpec.from_file("snippet.xml", vfs=vfs)
        frame = spec.worldbody.add_frame(name=frame_name)
        new_root_body = new_spec.body("pelvis") # Or the actual root body
        if new_root_body:
            frame.attach_body(new_root_body)
        
    return spec



