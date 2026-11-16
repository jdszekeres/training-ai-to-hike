import utils
import os
import math
import mujoco
import numpy as np
import rasterio
from scipy import ndimage
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

# --- Ground roughness ---------------------------------------------------------
# The DEM is far too coarse (~75-90 m cells) to have any human-scale detail, so
# the heightfield is resampled onto a finer grid and fractal noise is added.
HFIELD_TARGET_CELL_M = 1.0      # desired heightfield spacing (metres)
# Cap on nrow*ncol; large areas get coarser cells. Rendering cost scales with
# this (~0.6 s/frame at 1M on Intel UHD), physics barely notices it.
HFIELD_MAX_CELLS = 1_000_000
ROUGHNESS_RMS_AT_1M = 0.03      # RMS bump height (m) of 1 m-wavelength features
ROUGHNESS_HURST = 0.8           # amplitude grows as wavelength**HURST
ROUGHNESS_MAX_WAVELENGTH_M = 32.0
TRAIL_WIDTH_M = 2.0             # trails get much less roughness than open ground
TRAIL_ROUGHNESS_FACTOR = 0.2
ROUGHNESS_SEED = 0


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

def coordinate_to_pixel(lat, lon, terrain_file):
    """
    Converts geographic coordinates (latitude, longitude) to pixel coordinates in the terrain image.

    Parameters:
    lat: Latitude of the point.
    lon: Longitude of the point.
    terrain_file: Path to the terrain GeoTIFF file.

    Returns:
    A tuple (col, row) representing the pixel coordinates in the terrain image.
    """
    with rasterio.open(terrain_file) as dataset:
        # Get the affine transformation for the dataset
        transform = dataset.transform
        # Use the inverse of the transform to convert geographic coordinates to pixel coordinates
        inv_transform = ~transform
        col, row = inv_transform * (lon, lat)
        return int(col), int(row)

def pixel_to_world_coordinate(row, col, terrain_file):
    """
    Convert pixel coordiantes (row, col) to coordinates in the mujoco world frame (x, y)."""
    
    with rasterio.open(terrain_file) as dataset:
        transform = dataset.transform
        # Use the transform to convert pixel coordinates to geographic coordinates
        lon, lat = transform * (col + 0.5, row + 0.5)  # +0.5 to get the center of the pixel
        return lat, lon
    

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


def _hfieldHeight(hfield01, relief, half_x, half_y, x, y):
    """Bilinearly samples a MuJoCo hfield (row 0 = -y, normalized 0..1) at world (x, y)."""
    nrow, ncol = hfield01.shape
    c = np.clip((x + half_x) / (2 * half_x) * (ncol - 1), 0, ncol - 1)
    r = np.clip((y + half_y) / (2 * half_y) * (nrow - 1), 0, nrow - 1)
    c0, r0 = int(c), int(r)
    c1, r1 = min(c0 + 1, ncol - 1), min(r0 + 1, nrow - 1)
    fc, fr = c - c0, r - r0
    h = (hfield01[r0, c0] * (1 - fc) * (1 - fr) + hfield01[r0, c1] * fc * (1 - fr)
         + hfield01[r1, c0] * (1 - fc) * fr + hfield01[r1, c1] * fc * fr)
    return float(h) * relief


def _fractalRoughness(shape, cell_m, rng):
    """
    Zero-mean multi-octave noise (metres) on a grid with `cell_m` spacing.
    Octaves shorter than two cells can't be represented and are skipped.
    """
    noise = np.zeros(shape, dtype=np.float32)
    wavelength = max(1.0, 2.0 * cell_m)
    while wavelength <= ROUGHNESS_MAX_WAVELENGTH_M:
        octave = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32),
                                         sigma=wavelength / (4.0 * cell_m), mode="wrap")
        octave /= octave.std() + 1e-12
        noise += octave * (ROUGHNESS_RMS_AT_1M * wavelength ** ROUGHNESS_HURST)
        wavelength *= 2.0
    return noise


def buildWorldModel(lat1, lon1, lat2, lon2, name, add_person=True):
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

    # --- Fine heightfield with roughness -------------------------------------
    hf_cell_m = max(HFIELD_TARGET_CELL_M, math.sqrt(4 * half_x * half_y / HFIELD_MAX_CELLS))
    hf_ncol = int(round(2 * half_x / hf_cell_m)) + 1
    hf_nrow = int(round(2 * half_y / hf_cell_m)) + 1
    # Bicubic resample; zoom() aligns the first/last samples with the DEM's
    # first/last pixel centres, so the extent (half_x, half_y) is unchanged.
    fine_m = ndimage.zoom(elevation_m - minimum, (hf_nrow / nrow, hf_ncol / ncol),
                          order=3, mode="nearest", grid_mode=False).astype(np.float32)
    hf_nrow, hf_ncol = fine_m.shape

    # Trails at their real width, feathered, so they stay smoother than open ground.
    fine_trail_img = Image.new("L", (hf_ncol, hf_nrow), 0)
    fine_draw = ImageDraw.Draw(fine_trail_img)
    sx, sy = (hf_ncol - 1) / (ncol - 1), (hf_nrow - 1) / (nrow - 1)
    trail_px = max(1, int(round(TRAIL_WIDTH_M / hf_cell_m)))
    for pts in trail_lines:
        fine_draw.line([(c * sx, r * sy) for c, r in pts], fill=255, width=trail_px)
    fine_trail = ndimage.gaussian_filter(np.asarray(fine_trail_img, np.float32) / 255.0, sigma=1.0)
    fine_trail = np.clip(fine_trail / max(fine_trail.max(), 1e-6), 0.0, 1.0)

    rng = np.random.default_rng(ROUGHNESS_SEED)
    roughness = _fractalRoughness(fine_m.shape, hf_cell_m, rng)
    fine_m += roughness * (1.0 - (1.0 - TRAIL_ROUGHNESS_FACTOR) * fine_trail)
    print(f"Heightfield: {hf_ncol} x {hf_nrow} cells at {hf_cell_m:.2f} m, "
          f"roughness RMS {roughness.std():.2f} m")

    # MuJoCo stores heights normalized to 0..1 and scales them by size[2]; the
    # geom is offset by hf_lo so z stays "metres above the DEM minimum".
    hf_lo, hf_hi = float(fine_m.min()), float(fine_m.max())
    hf_relief = hf_hi - hf_lo
    # MuJoCo's hfield row 0 is at -y (south), GeoTIFF row 0 is north.
    hfield01 = np.flipud((fine_m - hf_lo) / hf_relief).astype(np.float32)
    hfield_name = name + "_hfield.bin"
    with open(os.path.join("data", hfield_name), "wb") as f:
        np.array([hf_nrow, hf_ncol], dtype=np.int32).tofile(f)
        hfield01.tofile(f)

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
        castshadow=False,
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
        file=hfield_name,  # binary: int32 nrow, int32 ncol, float32[nrow*ncol]
        size=[half_x, half_y, hf_relief, max(hf_relief * 0.05, 1.0)],
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
    terrain_geom.pos = np.array([0, 0, hf_lo], dtype=np.float64)

    # Rendering settings for a scene that is kilometres across.
    spec.stat.center = np.array([0, 0, relief / 2], dtype=np.float64)
    spec.stat.extent = math.hypot(half_x, half_y)
    spec.visual.global_.elevation = -60
    spec.visual.global_.azimuth = 90
    # znear/zfar are in units of stat.extent (kilometres here), not metres.
    # Keep the near plane at ~5 cm so a 1.3 m robot isn't clipped when zoomed in.
    spec.visual.map.znear = 0.05 / spec.stat.extent
    spec.visual.map.zfar = 3.0

    if add_person:
        # g1_mjx.xml uses only sphere/capsule/box colliders. MJX can't collide a
        # heightfield with cylinders (g1.xml's shoulders), and mesh colliders are slow.
        xml = open(os.path.join("robot", "g1_mjx.xml"), "r", encoding="utf-8").read()
        xml = xml.replace('<compiler angle="radian" meshdir="assets"/>', '<compiler angle="radian" meshdir="./robot/assets"/>') # Make sure assts are
        utils.import_mujoco_xml_to_spec(spec, xml, frame_name="person_frame")

        # Solver settings matching g1_mjx.xml (attaching a body doesn't copy <option>).
        spec.option.timestep = 0.004
        spec.option.iterations = 5
        spec.option.ls_iterations = 8
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP

        # g1_mjx.xml disables all automatic contacts (contype=conaffinity=0), so
        # every contact must be an explicit pair, as in robot/scene_mjx.xml.
        for side in ("left", "right"):
            for i in (1, 2, 3):
                spec.add_pair(name=f"{side}_foot{i}_terrain", geomname1=f"{side}_foot{i}_collision",
                              geomname2="terrain_geom", condim=3, solref=[0.008, 1],
                              friction=[1, 1, 0.005, 0.0001, 0.0001])
            for part in ("hand", "shoulder_yaw", "elbow_yaw", "wrist", "hip", "thigh", "shin"):
                spec.add_pair(name=f"{side}_{part}_terrain", geomname1=f"{side}_{part}_collision",
                              geomname2="terrain_geom", condim=3)
        for part in ("pelvis", "torso", "head"):
            spec.add_pair(name=f"{part}_terrain", geomname1=f"{part}_collision",
                          geomname2="terrain_geom", condim=3)

        # Self-collision pairs from robot/scene_mjx.xml.
        self_pairs = [
            ("left_foot_box", "right_foot_box"),
            ("left_foot_box", "right_shin"), ("right_foot_box", "left_shin"),
            ("left_foot_box", "right_linkage_brace"), ("right_foot_box", "left_linkage_brace"),
            ("left_hand", "left_hip"), ("right_hand", "right_hip"),
            ("left_hand", "left_thigh"), ("right_hand", "right_thigh"),
            ("left_shin", "right_shin"),
            ("torso", "left_shoulder_yaw"), ("torso", "right_shoulder_yaw"),
            ("torso", "left_elbow_yaw"), ("torso", "right_elbow_yaw"),
            ("torso", "left_wrist"), ("torso", "right_wrist"),
            ("torso", "left_hand"), ("torso", "right_hand"),
            ("left_thigh", "right_thigh"),
            ("left_shin", "right_thigh"), ("right_shin", "left_thigh"),
            ("left_shin", "right_hip"), ("right_shin", "left_hip"),
            ("left_hip", "right_thigh"), ("right_hip", "left_thigh"),
            ("left_hand", "right_hand"),
        ]
        for a, b in self_pairs:
            spec.add_pair(name=f"{a}_{b}", geomname1=f"{a}_collision",
                          geomname2=f"{b}_collision", condim=1)

        # Spawn just above the actual ground at (0, 0) instead of above the
        # highest peak, so the robot doesn't free-fall hundreds of metres.
        ground_z = hf_lo + _hfieldHeight(hfield01, hf_relief, half_x, half_y, 0.0, 0.0)
        pelvis = spec.body("pelvis")
        pelvis.pos = np.array([0, 0, ground_z + 0.8], dtype=np.float64)  # g1 pelvis stands ~0.79 m high

        # Chase camera; select it in the viewer's Rendering > Camera list (or press ']').
        pelvis.add_camera(
            name="follow",
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
            pos=[-10, 0, 3],
            xyaxes=[0, -1, 0, 0.3, 0, 1],
            quat=[-1, 0, 0, -1]
        )
    model = spec.compile()
    with open(os.path.join("data", name + ".xml"), "w", encoding="utf-8") as f:
        f.write(spec.to_xml())

    return (model, spec)
