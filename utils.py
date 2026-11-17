import math
import time
from mujoco import mujoco
from mujoco._structs import MjModel
import rasterio
import requests
import os

# Approximate metres per degree of latitude/longitude (WGS84).
M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQUATOR = 111320.0

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





class CoordinateSystem:
    """
    A class to represent coordinates betweeen all the different types in this project. This is:
    - Latitude/Longitude (WGS84)
    - Pixel coordinates (row, col)
    - World coordinates (x, y)
    """
    def __init__(self: 'CoordinateSystem', lat: float, lon: float):
        self.lat = lat
        self.lon = lon
        

    @staticmethod
    def from_lat_lon(lat: float, lon: float) -> 'CoordinateSystem':
        return CoordinateSystem(lat, lon)

    @staticmethod
    def from_pixel(row: int, col: int, model: MjModel, bbox: tuple) -> 'CoordinateSystem':
        nrow, ncol = model.hfield_nrow[0], model.hfield_ncol[0]
        # Convert pixel coorinates to lat lon
        lat = bbox[1] + (bbox[3] - bbox[1]) * (row / nrow)
        lon = bbox[0] + (bbox[2] - bbox[0]) * (col / ncol)
        
        return CoordinateSystem(lat, lon)

    @staticmethod
    def _world_grid(terrain_file: str):
        """
        (transform, nrow, ncol, dx_m, dy_m) describing how terrain.buildWorldModel
        lays the DEM out in the world: pixel centres span [-half, +half] metres,
        centred on (0, 0), with +x = east and +y = north.
        """
        with rasterio.open(terrain_file) as dataset:
            transform = dataset.transform
            nrow, ncol = dataset.height, dataset.width
        lat_mid = transform.f + transform.e * nrow / 2
        dx_m = abs(transform.a) * M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat_mid))
        dy_m = abs(transform.e) * M_PER_DEG_LAT
        return transform, nrow, ncol, dx_m, dy_m

    @staticmethod
    def from_world(x: float, y: float, terrain_file: str) -> 'CoordinateSystem':
        transform, nrow, ncol, dx_m, dy_m = CoordinateSystem._world_grid(terrain_file)
        col = x / dx_m + (ncol - 1) / 2   # pixel-centre index, 0 = west edge
        row = (nrow - 1) / 2 - y / dy_m   # pixel-centre index, 0 = north edge
        lon, lat = transform * (col + 0.5, row + 0.5)  # transform expects pixel corners
        return CoordinateSystem(lat, lon)

    def to_lat_lon(self) -> tuple:
        return (self.lat, self.lon)

    def to_pixel(self, model: MjModel, BOUNDING_BOX: tuple) -> tuple:
        nrow, ncol = model.hfield_nrow[0], model.hfield_ncol[0]
        row = int((self.lat - BOUNDING_BOX[1]) / (BOUNDING_BOX[3] - BOUNDING_BOX[1]) * nrow)
        col = int((self.lon - BOUNDING_BOX[0]) / (BOUNDING_BOX[2] - BOUNDING_BOX[0]) * ncol)
        return (row, col)

    def to_world(self, terrain_file: str) -> tuple[float, float]:
        """World (x, y) in metres, matching the heightfield built by terrain.buildWorldModel."""
        transform, nrow, ncol, dx_m, dy_m = CoordinateSystem._world_grid(terrain_file)
        # ~transform maps (lon, lat) -> (col, row) of pixel corners; -0.5 gives pixel centres.
        col, row = ~transform * (self.lon, self.lat)
        col, row = col - 0.5, row - 0.5
        x = (col - (ncol - 1) / 2) * dx_m
        y = ((nrow - 1) / 2 - row) * dy_m  # GeoTIFF row 0 is north (+y)
        return (float(x), float(y))


    