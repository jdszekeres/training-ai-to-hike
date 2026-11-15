import utils
import os
import math
import mujoco
import numpy as np
import rasterio
from PIL import Image, ImageDraw
import json

# Approximate metres per degree of latitude/longitude (WGS84).
M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQUATOR = 111320.0

# Longest side (in pixels) of the generated terrain texture. The DEM itself is
# coarse (~90 m/pixel for SRTMGL3), so the texture is upsampled so trails can be
# drawn as thin, continuous lines instead of blocky single pixels.
TEXTURE_MAX_SIZE = 4096

# Hypsometric colour ramp: (normalized height, RGB)
TERRAIN_COLORS = [
    (0.00, (40, 95, 45)),     # low valleys / forest
    (0.25, (85, 135, 60)),    # meadows
    (0.45, (165, 160, 95)),   # dry grass / scrub
    (0.65, (140, 110, 80)),   # bare earth
    (0.85, (130, 125, 120)),  # rock
    (1.00, (245, 245, 250)),  # snow
]
TRAIL_COLOR = (230, 30, 30)


def fetchOpenTopography(key, lat1, lon1, lat2, lon2, name):
    """
    Fetches elevation data from OpenTopography for the specified bounding box and saves it to a file.

    Parameters:
    lat1, lon1: Coordinates of the southwest corner of the bounding box.
    lat2, lon2: Coordinates of the northeast corner of the bounding box.
    name: Name of the area (used for the output file name).
    """
    if os.path.exists(os.path.join("data", name+".tif")):
        print(f"File {name}.tif already exists. Skipping download.")
        return os.path.join("data", name+".tif")

    if not key:
        raise ValueError("API key is required")
    url = f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL3&south={lat1}&north={lat2}&west={lon1}&east={lon2}&outputFormat=GTiff&API_Key={key}"

    # Use utils to download the data
    utils.download_file(url, name+".tif")

    return os.path.join("data", name+".tif")


def getTrails(lat1, lon1, lat2, lon2, cache_file=None):
    """
    Fetches trail data from OpenStreetMap for the specified bounding box.

    Parameters:
    lat1, lon1: Coordinates of the southwest corner of the bounding box.
    lat2, lon2: Coordinates of the northeast corner of the bounding box.
    cache_file: Optional path to cache the response in, so Overpass is only queried once.

    Returns:
    The JSON response from the OpenStreetMap API.
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached trails from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    query = f"""
    [out:json];
    (
      way["highway"="path"]({lat1},{lon1},{lat2},{lon2});
      way["highway"="footway"]({lat1},{lon1},{lat2},{lon2});
      way["highway"="track"]({lat1},{lon1},{lat2},{lon2});
    );
    out geom;
    """
    json_response = utils.overpass_query(query)

    if cache_file:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(json_response, f)
    return json_response


def _trailPolylines(trails_data, inv_transform):
    """Converts OSM ways into lists of fractional (col, row) DEM pixel-centre coordinates."""
    lines = []
    for element in trails_data.get('elements', []):
        if element.get('type') != 'way' or 'geometry' not in element:
            continue
        pts = []
        for point in element['geometry']:
            # ~transform maps (lon, lat) -> (col, row) of pixel *corners*;
            # subtract 0.5 so integer values land on pixel centres.
            col, row = inv_transform * (point['lon'], point['lat'])
            pts.append((col - 0.5, row - 0.5))
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def _colorize(height01):
    """Maps a normalized [0, 1] height array to RGB using TERRAIN_COLORS."""
    stops = np.array([s for s, _ in TERRAIN_COLORS])
    colors = np.array([c for _, c in TERRAIN_COLORS], dtype=np.float32)
    rgb = np.empty(height01.shape + (3,), dtype=np.float32)
    for ch in range(3):
        rgb[..., ch] = np.interp(height01, stops, colors[:, ch])
    return rgb


def _hillshade(elev_m, dx_m, dy_m, azimuth_deg=315.0, altitude_deg=45.0):
    """Classic hillshade in [0, 1]. Rows run north->south, columns west->east."""
    dz_drow, dz_dcol = np.gradient(elev_m, dy_m, dx_m)
    dz_dx = dz_dcol     # east is +x
    dz_dy = -dz_drow    # rows increase southwards, so flip for north = +y
    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)
    # Light direction (pointing towards the sun)
    lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    nz = 1.0 / np.sqrt(1.0 + dz_dx ** 2 + dz_dy ** 2)
    shade = (-dz_dx * lx - dz_dy * ly + lz) * nz
    return np.clip(shade, 0.0, 1.0)


def buildWorldFromElevationData(lat1, lon1, lat2, lon2, name):
    """
    Builds a world representation from the elevation data file.

    Parameters:
    lat1, lon1: Coordinates of the southwest corner of the bounding box.
    lat2, lon2: Coordinates of the northeast corner of the bounding box.
    name: Name of the area.

    Returns:
    A compiled MuJoCo model of the terrain.
    """
    OPENTOPO_API_KEY = os.environ.get("OPENTOPO_API_KEY")
    path = fetchOpenTopography(OPENTOPO_API_KEY, lat1, lon1, lat2, lon2, name)

    trails_data = getTrails(lat1, lon1, lat2, lon2,
                            cache_file=os.path.join("data", name + "_trails.json"))

    with rasterio.open(path) as dataset:
        elevation_data = dataset.read(1).astype(np.float32)
        nodata = dataset.nodata
        transform = dataset.transform

    nrow, ncol = elevation_data.shape

    # Handle missing data
    if nodata is not None:
        elevation_data[elevation_data == nodata] = np.nan
    valid = np.isfinite(elevation_data)

    minimum = float(elevation_data[valid].min())
    maximum = float(elevation_data[valid].max())
    print(f"Elevation data range: min={minimum}, max={maximum}")

    # Fill missing data with the minimum so the heightfield has no holes/spikes.
    elevation_m = np.where(valid, elevation_data, minimum).astype(np.float32)

    # Real-world pixel size in metres (the GeoTIFF is in degrees).
    lat_mid = transform.f + transform.e * nrow / 2
    dx_m = abs(transform.a) * M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat_mid))
    dy_m = abs(transform.e) * M_PER_DEG_LAT
    # Heightfield vertices sit on pixel centres, so the span is (n - 1) pixels.
    half_x = (ncol - 1) * dx_m / 2
    half_y = (nrow - 1) * dy_m / 2
    relief = maximum - minimum if maximum > minimum else 1.0
    print(f"Terrain extent: {2*half_x:.0f} m x {2*half_y:.0f} m, relief {relief:.0f} m")

    # --- Trails --------------------------------------------------------------
    trail_lines = _trailPolylines(trails_data, ~transform)

    # Rasterize trails at DEM resolution and smooth the terrain under them.
    trail_mask_img = Image.new("L", (ncol, nrow), 0)
    mask_draw = ImageDraw.Draw(trail_mask_img)
    for pts in trail_lines:
        mask_draw.line(pts, fill=255, width=1)
    trail_mask = np.array(trail_mask_img) > 0

    padded = np.pad(elevation_m, 1, mode="edge")
    blurred = sum(padded[1 + dy:1 + dy + nrow, 1 + dx:1 + dx + ncol]
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    elevation_m[trail_mask] = blurred[trail_mask]

    # Normalized heights for MuJoCo (0..1, scaled by size[2]).
    elevation = np.clip((elevation_m - minimum) / relief, 0.0, 1.0)

    # --- Texture -------------------------------------------------------------
    scale = max(1, TEXTURE_MAX_SIZE // max(ncol, nrow))
    tex_w, tex_h = (ncol - 1) * scale + 1, (nrow - 1) * scale + 1

    hi_elev = np.array(Image.fromarray(elevation_m, mode="F").resize((tex_w, tex_h), Image.BICUBIC))
    hi_valid = np.array(Image.fromarray(valid.astype(np.uint8) * 255).resize((tex_w, tex_h), Image.NEAREST)) > 0

    rgb = _colorize(np.clip((hi_elev - minimum) / relief, 0.0, 1.0))
    shade = _hillshade(hi_elev, dx_m / scale, dy_m / scale)
    rgb *= (0.45 + 0.75 * shade)[..., None]
    rgb[~hi_valid] = 0  # Black for missing data
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(img)
    trail_width = max(2, scale // 2)
    for pts in trail_lines:
        draw.line([(c * scale, r * scale) for c, r in pts],
                  fill=TRAIL_COLOR, width=trail_width, joint="curve")

    texture_name = name + "_terrain.png"
    img.save(os.path.join("data", texture_name))

    # Create mujoco world model
    spec = mujoco.MjSpec()
    spec.modelname = "terrain_model"
    # Resolve asset paths relative to the data folder, so the same relative
    # file name works both when compiling now and when loading data/<name>.xml.
    spec.modelfiledir = os.path.abspath("data")

    spec.worldbody.add_light(
        name="sun",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0, 0, relief * 2], # At least twice as high as our terrain
        dir=[0.3, 0.3, -1],
        diffuse=[0.8, 0.8, 0.8],
        ambient=[0.35, 0.35, 0.35],
        specular=[0, 0, 0],
        castshadow=True,
    )

    # Top-down camera high enough that the entire heightfield fits in view.
    fovy = 45.0
    cam_height = relief + 1.1 * max(half_x, half_y) / math.tan(math.radians(fovy / 2))
    spec.worldbody.add_camera(
        name="main_camera",
        pos=[0, 0, cam_height],
        xyaxes=[1, 0, 0, 0, 1, 0],
        fovy=fovy,
    )

    spec.add_hfield(
        name="terrain",
        nrow=nrow,
        ncol=ncol,
        size=[half_x, half_y, relief, max(relief * 0.05, 1.0)],
        # MuJoCo's hfield row 0 is at -y (south), GeoTIFF row 0 is north.
        userdata=np.flipud(elevation).flatten().tolist(),
    )

    spec.add_texture(
        name="terrain_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=texture_name,
        colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_SRGB,
    )
    spec.add_material(
        name="terrain_material",
        textures=["", "terrain_texture"],  # slot 1 = mjTEXROLE_RGB
        texuniform=False,   # stretch the texture once over the whole hfield
        texrepeat=[1, 1],
        specular=0.0,
        shininess=0.0,
    )

    terrain_geom = spec.worldbody.add_geom()
    terrain_geom.name = "terrain_geom"
    terrain_geom.type = mujoco.mjtGeom.mjGEOM_HFIELD
    terrain_geom.hfieldname = "terrain"
    terrain_geom.material = "terrain_material"

    # Rendering settings for a scene that is kilometres across.
    spec.stat.center = [0, 0, relief / 2]
    spec.stat.extent = math.hypot(half_x, half_y)
    spec.visual.global_.elevation = -60
    spec.visual.global_.azimuth = 90
    spec.visual.map.znear = 0.001
    spec.visual.map.zfar = 20.0

    model = spec.compile()
    with open(os.path.join("data", name + ".xml"), "w", encoding="utf-8") as f:
        f.write(spec.to_xml())

    return model
