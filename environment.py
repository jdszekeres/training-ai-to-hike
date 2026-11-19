"""
Gymnasium environment: G1 humanoid on a DEM heightfield, simulated with
CPU MuJoCo so it can be stepped by Stable-Baselines3.

The observation, reward and termination match the previous MJX TerrainEnv
(kept in legacy/environment_mjx.py).
"""
import math
import time

import gymnasium as gym
import mujoco
from mujoco._structs import MjModel
import numpy as np

from utils import CoordinateSystem

# Joint angles of the "home" keyframe in robot/scene_mjx.xml (qpos[7:], which
# also equals ctrl since every actuator is a position servo on one joint).
# Attaching g1_mjx.xml to the terrain spec doesn't carry keyframes over.
G1_HOME_JOINTS = np.array([
    -0.1, 0, 0, 0.3, -0.2, 0,          # left leg
    -0.1, 0, 0, 0.3, -0.2, 0,          # right leg
    0, 0, 0,                           # waist
    0.2, 0.2, 0, 1.28, 0, 0, 0,        # left arm
    0.2, -0.2, 0, 1.28, 0, 0, 0,       # right arm
], dtype=np.float64)


class HField:
  """Heightfield grid (normalized 0..1, row 0 = -y) plus its world placement."""

  def __init__(self, mj_model):
    nrow, ncol = int(mj_model.hfield_nrow[0]), int(mj_model.hfield_ncol[0])
    adr = int(mj_model.hfield_adr[0])
    self.data = np.asarray(mj_model.hfield_data[adr:adr + nrow * ncol]).reshape(nrow, ncol)
    # size = (half_x, half_y, relief, base) as set in terrain.buildWorldModel.
    self.half_x, self.half_y, self.relief = (float(v) for v in mj_model.hfield_size[0, :3])
    geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_geom")
    if geom_id < 0:  # fall back to whichever geom uses hfield 0
      geom_id = int(np.flatnonzero(
          (mj_model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD) & (mj_model.geom_dataid == 0))[0])
    self.origin = np.array(mj_model.geom_pos[geom_id], dtype=np.float64)

  def height(self, x, y):
    """World-frame ground z at world (x, y); bilinear, clamped to the field's edge."""
    nrow, ncol = self.data.shape
    x, y = x - self.origin[0], y - self.origin[1]
    c = min(max((x + self.half_x) / (2 * self.half_x) * (ncol - 1), 0.0), ncol - 1)
    r = min(max((y + self.half_y) / (2 * self.half_y) * (nrow - 1), 0.0), nrow - 1)
    c0, r0 = int(c), int(r)
    c1, r1 = min(c0 + 1, ncol - 1), min(r0 + 1, nrow - 1)
    fc, fr = c - c0, r - r0
    hf = self.data
    h = (hf[r0, c0] * (1 - fc) * (1 - fr) + hf[r0, c1] * fc * (1 - fr)
         + hf[r1, c0] * (1 - fc) * fr + hf[r1, c1] * fc * fr)
    return float(self.origin[2] + h * self.relief)



def get_terrain_height(model: MjModel, coordinates: CoordinateSystem, terrain_file: str) -> float:
  """Terrain height at world (x, y) for a compiled MjModel (0 if it has no heightfield)."""
  return HField(model).height(
    *coordinates.to_world(terrain_file)
  ) if model.nhfield else 0.0


class TerrainEnv(gym.Env):
  """
  Actions are in [-1, 1] and are offsets from the home pose, scaled by
  `action_scale` radians and clipped to each actuator's ctrlrange.

  preview=True (or render_mode="human") opens a passive MuJoCo viewer that
  follows the robot and is refreshed (at most `render_fps` times per second of
  wall-clock time) from inside step(), so a training run can be watched live.
  It is a separate flag from render_mode because SubprocVecEnv requires every
  worker to share one render_mode, while only one of them should open a window.
  """

  metadata = {"render_modes": ["human"], "render_fps": 30}

  def __init__(self, model, start_position=(0,0), goal_position=None, step_dt=0.02, action_scale=0.5,
               reset_noise=0.05, render_mode=None, preview=False):
    # MjModel is picklable, so this env can be built inside SubprocVecEnv workers.
    self.model = model

    self.start_position = np.asarray(start_position, dtype=np.float64)

    self.prev_position = tuple(start_position)

    self.data = mujoco.MjData(model)
    self.goal_position = np.asarray(
        goal_position if goal_position is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    self._hfield = HField(model) if model.nhfield else None

    # Control timing: one env step = n_frames physics steps.
    self.n_frames = max(1, int(round(step_dt / model.opt.timestep)))
    self.dt = self.n_frames * model.opt.timestep

    # Initial pose: free joint from qpos0 (spawn point set in terrain.py),
    # joints from the home keyframe.
    self._init_qpos = model.qpos0.copy()
    if model.nq - 7 == len(G1_HOME_JOINTS):
      self._init_qpos[7:] = G1_HOME_JOINTS
    self._default_ctrl = self._init_qpos[7:7 + model.nu].copy()
    self._ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    self._ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
    self.action_scale = action_scale
    self.reset_noise = reset_noise

    self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(model.nu,), dtype=np.float32)
    self._reset_sim()
    obs_size = self._get_observation().shape[0]
    self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    self.preview = preview or render_mode == "human"
    self._viewer = None
    self._viewer_closed = False
    self._last_sync = 0.0

  # --- Terrain ------------------------------------------------------------------
  def get_terrain_height(self, x, y):
    return self._hfield.height(x, y) if self._hfield is not None else 0.0

  # --- MDP ----------------------------------------------------------------------
  def _get_observation(self):
    d = self.data

    sensors =self.get_sensors()  # update d.sensordata

    return np.concatenate([
        d.qpos,                 # positions
        d.qvel,                 # velocities
        d.xpos.ravel(),         # body positions
        [self.get_terrain_height(d.qpos[0], d.qpos[1])],  # height at robot's position
        self.goal_position - d.qpos[:3],                   # pelvis -> goal vector

        sensors["imu-torso-angular-velocity"].flatten(),
        sensors["imu-torso-linear-acceleration"].flatten(),
        sensors["imu-pelvis-angular-velocity"].flatten(),
        sensors["imu-pelvis-linear-acceleration"].flatten(),

        self.prev_position - d.qpos[:2],  # previous position -> current position vector
    ]).astype(np.float32)


  def _get_sensor_vals(self, sensor_name):
    sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    adr, dim = self.model.sensor_adr[sid], self.model.sensor_dim[sid]
    return self.data.sensordata[adr:adr + dim]
  def get_sensors(self):
    d = self.data
    
    return {
        "imu-torso-angular-velocity": self._get_sensor_vals("imu-torso-angular-velocity"),
        "imu-torso-linear-acceleration": self._get_sensor_vals("imu-torso-linear-acceleration"),
        "imu-pelvis-angular-velocity": self._get_sensor_vals("imu-pelvis-angular-velocity"),
        "imu-pelvis-linear-acceleration": self._get_sensor_vals("imu-pelvis-linear-acceleration")
    }


  def _RBF_kernel(self, x, x_prime, sigma=1.0):
    """Radial Basis Function (RBF) kernel."""
    return np.exp(-np.linalg.norm(x - x_prime) ** 2 / (2 * sigma ** 2))

  def _calculate_terrain_gradient(self, x, y, delta=0.01):
    """Calculate the gradient of the terrain at a given (x, y) position."""
    h = self.get_terrain_height(x, y)
    h_x_plus = self.get_terrain_height(x + delta, y)
    h_x_minus = self.get_terrain_height(x - delta, y)
    h_y_plus = self.get_terrain_height(x, y + delta)
    h_y_minus = self.get_terrain_height(x, y - delta)

    dh_dx = (h_x_plus - h_x_minus) / (2 * delta)
    dh_dy = (h_y_plus - h_y_minus) / (2 * delta)

    return np.array([dh_dx, dh_dy])

  def _get_reward(self, action):
    # Distance to the goal, energy efficiency, and stability.
    d = self.data

    h = d.qpos[2] - self.get_terrain_height(d.qpos[0], d.qpos[1])
    sqrt2 = math.sqrt(2)
    is_fallen = h < 0.3  # If the pelvis height is below 0.3 meters, consider it a fall or the torso is not upright

    alive_bonus = 1 if not is_fallen else 0
    upright_bonus = 0 # TODO: Add upright bonus based on torso orientation
    pelvis_height_score = self._RBF_kernel(h, 0.75, 0.25 / sqrt2)  # Encourage pelvis height around 0.75m



    angular_velocity_penalty = -np.linalg.norm(d.qvel[3:5]) ** 2  # Penalize high angular velocity
    vertical_velocity_penalty = -np.linalg.norm(d.qvel[2]) ** 2  # Penalize high vertical velocity

    action_efficiency_penalty = -np.sum(np.square(action))  # Penalize large actions
    
    power_penalty = -np.sum(np.abs(d.actuator_force * d.actuator_velocity))  # Penalize high power usage
    
    torque_penalty = -np.sum(np.square(d.actuator_force))  # Penalize high actuator forces

    arm_penalty = -np.sum(np.square(d.qpos[19:26]))  # Penalize large arm movements

    terrain_gradient = self._calculate_terrain_gradient(d.qpos[0], d.qpos[1])
    slope = np.linalg.norm(terrain_gradient)
    slope_penalty = -slope

    expected_speed = 1000 * (1/60) *  6 * math.exp(-3.5 * abs(slope + 0.05)) # Tobler's hiking function in m/s

    speed_score = self._RBF_kernel(np.linalg.norm(d.qvel[0:3]), expected_speed, 1.5 / sqrt2)  # Encourage speed close to expected speed
    
    progress_bonus = np.linalg.norm(self.prev_position - d.qpos[:2]) / self.dt  # Reward for moving forward towards the goal

    self.prev_position = d.qpos[:2].copy()  # Update previous position for the next step

    
    return (
        1.0 * progress_bonus +  # Encourage forward progress
        0.5 * upright_bonus +  # Encourage upright posture
        0.2 * pelvis_height_score +  # Encourage pelvis height around 0.75m
        0.5 * speed_score +  # Encourage speed close to expected speed
        0.5 * alive_bonus +  # Penalize falling
        0.1 * angular_velocity_penalty +  # Penalize high angular velocity
        0.1 * vertical_velocity_penalty +  # Penalize high vertical velocity
        0.01 * action_efficiency_penalty +  # Penalize large actions
        0.0002 * power_penalty +  # Penalize high power usage
        0.0001 * torque_penalty +  # Penalize high actuator forces
        0.05 * arm_penalty +  # Penalize large arm movements
        0.1 * slope_penalty   # Penalize steep slopes
    )

  def _get_done(self):
    return bool(np.linalg.norm(self.data.qpos[:3] - self.goal_position) < 0.1) or self.data.qpos[2] - self.get_terrain_height(self.data.qpos[0], self.data.qpos[1]) < 0.5  # Terminate if the robot is close to the goal or has fallen

  def _reset_sim(self):
    mujoco.mj_resetData(self.model, self.data)
    qpos = self._init_qpos.copy()
    qpos[:3] = [*self.start_position, self.get_terrain_height(*self.start_position) + 0.8]  # Start 0.5m above the terrain
    qpos[7:] += self.np_random.uniform(-self.reset_noise, self.reset_noise, self.model.nq - 7)
    self.data.qpos[:] = qpos
    self.data.qvel[:] = 0.0
    self.data.ctrl[:] = self._default_ctrl
    mujoco.mj_forward(self.model, self.data)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self._reset_sim()
    if self.preview:
      self._draw(force=True)
    return self._get_observation(), {}

  def step(self, action):
    action = np.asarray(action, dtype=np.float64)
    self.data.ctrl[:] = np.clip(self._default_ctrl + self.action_scale * action,
                                self._ctrl_lo, self._ctrl_hi)
    mujoco.mj_step(self.model, self.data, nstep=self.n_frames)

    obs = self._get_observation()
    reward = self._get_reward(action)
    terminated = self._get_done()
    # A diverged simulation can't be recovered; end the episode instead of feeding NaNs to PPO.
    if not np.all(np.isfinite(obs)):
      obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
      reward, terminated = 0.0, True
    if self.preview:
      self._draw()
    return obs, reward, terminated, False, {}

  # --- Rendering ----------------------------------------------------------------
  def render(self):
    if self.render_mode == "human":
      self._draw(force=True)

  def _draw(self, force=False):
    if self._viewer_closed:
      return
    now = time.perf_counter()
    if not force and now - self._last_sync < 1.0 / self.metadata["render_fps"]:
      return
    self._last_sync = now
    if self._viewer is None:
      # Imported lazily so headless machines (e.g. Google Colab) never need a display.
      import mujoco.viewer
      self._viewer = mujoco.viewer.launch_passive(
          self.model, self.data, show_left_ui=False, show_right_ui=False)
      self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      self._viewer.cam.trackbodyid = self.model.body("pelvis").id
      self._viewer.cam.distance = 4
      self._viewer.cam.elevation = -20
    if not self._viewer.is_running():
      # The user closed the window: keep training, just stop drawing.
      self._viewer_closed = True
      self._viewer = None
      return
    self._viewer.sync()

  def close(self):
    if self._viewer is not None:
      self._viewer.close()
      self._viewer = None
