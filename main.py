import os
import time

import dotenv
import mujoco
import mujoco.viewer
import numpy as np
import terrain
import environment
import ppo

dotenv.load_dotenv()
if not os.path.exists("data"):
    os.makedirs("data")


OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY")


if __name__ == "__main__":
    # This is the WGS84 bounding box format: (lon1, lat1, lon2, lat2)
    BOUNDING_BOX = (-118.392068, 36.534354, -118.228898, 36.616395) # Mount Whitney area

    GOAL = (36.5785, -118.292) # Whiney summit

    NAME="WHITNEY"
    SAVE_PATH = os.path.join("checkpoints", "ppo_hike")

    (model, spec) = terrain.buildWorldModel(BOUNDING_BOX[1], BOUNDING_BOX[0], BOUNDING_BOX[3], BOUNDING_BOX[2], NAME)

    goal_col, goal_row = terrain.coordinate_to_pixel(GOAL[0], GOAL[1], os.path.join("data", NAME + ".tif"))
    goal_world_x, goal_world_y = terrain.pixel_to_world_coordinate(goal_row, goal_col, os.path.join("data", NAME + ".tif"))
    goal_height = environment.get_terrain_height(model, goal_world_x, goal_world_y)
    goal_position = np.array([goal_world_x, goal_world_y, goal_height])

    # Stable-Baselines3 PPO on CPU MuJoCo. preview=True opens a viewer on the
    # first worker so you can watch a robot while it trains (Ctrl+C stops
    # training early and still saves the policy).
    ppo.train(
        model,
        goal_position,
        total_timesteps=5_000_000,
        episode_length=1000,
        preview=True,
        save_path=SAVE_PATH,
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
    )

    # Watch the trained policy.
    policy = ppo.load_policy(SAVE_PATH)
    env = environment.TerrainEnv(model, goal_position=goal_position)
    obs, _ = env.reset(seed=0)
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = model.body("pelvis").id
        v.cam.distance = 4
        v.cam.elevation = -20
        while v.is_running():
            t0 = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            if terminated or truncated:
                obs, _ = env.reset()
            v.sync()
            time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))  # real time
