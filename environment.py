"""
Gymnasium environment: G1 humanoid on a DEM heightfield, simulated with
CPU MuJoCo so it can be stepped by Stable-Baselines3.

The observation, reward and termination match the previous MJX TerrainEnv
(kept in legacy/environment_mjx.py).
"""
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

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


def get_terrain_height(model, x, y):
  """Terrain height at world (x, y) for a compiled MjModel (0 if it has no heightfield)."""
  return HField(model).height(x, y) if model.nhfield else 0.0


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

  def __init__(self, model, goal_position=None, step_dt=0.02, action_scale=0.5,
               reset_noise=0.05, render_mode=None, preview=False):
    # MjModel is picklable, so this env can be built inside SubprocVecEnv workers.
    self.model = model
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
    return np.concatenate([
        d.qpos,                 # positions
        d.qvel,                 # velocities
        d.act,                  # actuator states
        d.xpos.ravel(),         # body positions
        [self.get_terrain_height(d.qpos[0], d.qpos[1])],  # height at robot's position
        self.goal_position - d.qpos[:3],                   # pelvis -> goal vector
    ]).astype(np.float32)

  def _get_reward(self, action):
    # Distance to the goal, energy efficiency, and stability.
    d = self.data
    distance_reward = -np.linalg.norm(d.qpos[:3] - self.goal_position)
    energy_reward = -np.sum(np.square(action))      # Penalize large actions
    stability_reward = -np.sum(np.square(d.qvel))   # Penalize high velocities

    #If fall, end the episode and give a large negative reward
    if d.qpos[2] < 0.5:  # If the pelvis height is below 0.5 meters, consider it a fall
        return -100.0  # Large negative reward for falling
    
    
    return float(distance_reward + 0.1 * energy_reward + 0.1 * stability_reward)

  def _get_done(self):
    return bool(np.linalg.norm(self.data.qpos[:3] - self.goal_position) < 0.1) or self.data.qpos[2] < 0.5  # Terminate if the robot is close to the goal or has fallen

  def _reset_sim(self):
    mujoco.mj_resetData(self.model, self.data)
    qpos = self._init_qpos.copy()
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
