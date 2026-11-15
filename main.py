import os
import dotenv
import mujoco
import mujoco.viewer
import terrain


dotenv.load_dotenv()
if not os.path.exists("data"):
    os.makedirs("data")


OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY")


if __name__ == "__main__":
    # This is the WGS84 bounding box format: (lon1, lat1, lon2, lat2)
    BOUNDING_BOX = (-118.392068, 36.534354, -118.228898, 36.616395) # Mount Whitney area
    NAME="WHITNEY"

    model = terrain.buildWorldFromElevationData(BOUNDING_BOX[1], BOUNDING_BOX[0], BOUNDING_BOX[3], BOUNDING_BOX[2], NAME)
    data = mujoco.MjData(model)
    
    # View the world
    
    mujoco.viewer.launch(model, data)
    
    


