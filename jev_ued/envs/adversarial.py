# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An environment which is built by a learning adversary.

Has additional functions, step_adversary, and reset_agent. How to use:
1. Call reset() to reset to an empty environment
2. Call step_adversary() to place the goal, agent, and obstacles. Repeat until
   a done is received.
3. Normal RL loop. Use learning agent to generate actions and use them to call
   step() until a done is received.
4. If required, call reset_agent() to reset the environment the way the
   adversary designed it. A new agent can now play it using the step() function.

When created through gymnasium.make(), use env.unwrapped to reach
step_adversary() and the other adversary methods.
"""
import gymnasium as gym
import networkx as nx
from networkx import grid_graph
import numpy as np
from minigrid.core.world_object import Goal, Wall

from . import multigrid


class AdversarialEnv(multigrid.MultiGridEnv):
  """Grid world where an adversary build the environment the agent plays.

  The adversary places the goal, agent, and up to n_clutter blocks in sequence.
  The action dimension is the number of squares in the grid, and each action
  chooses where the next item should be placed.
  """

  def __init__(self, n_clutter=50, size=15, agent_view_size=5, max_steps=250,
               goal_noise=0., random_z_dim=50, choose_goal_last=False,
               seed=0, fixed_environment=False, see_through_walls=True,
               render_mode='rgb_array'):
    """Initializes environment in which adversary places goal, agent, obstacles.

    Args:
      n_clutter: The maximum number of obstacles the adversary can place.
      size: The number of tiles across one side of the grid; i.e. make a
        size x size grid.
      agent_view_size: The number of tiles in one side of the agent's partially
        observed view of the grid.
      max_steps: The maximum number of steps that can be taken before the
        episode terminates.
      goal_noise: The probability with which the goal will move to a different
        location than the one chosen by the adversary.
      random_z_dim: The environment generates a random vector z to condition the
        adversary. This gives the dimension of that vector.
      choose_goal_last: If True, will place the goal and agent as the last
        actions, rather than the first actions.
    """
    self.agent_start_pos = None
    self.goal_pos = None
    self.n_clutter = n_clutter
    self.goal_noise = goal_noise
    self.random_z_dim = random_z_dim
    self.choose_goal_last = choose_goal_last

    # Add two actions for placing the agent and goal.
    self.adversary_max_steps = self.n_clutter + 2

    super().__init__(
        n_agents=1,
        minigrid_mode=True,
        grid_size=size,
        max_steps=max_steps,
        agent_view_size=agent_view_size,
        see_through_walls=see_through_walls,  # Set this to True for maximum speed
        competitive=True,
        seed=seed,
        fixed_environment=fixed_environment,
        render_mode=render_mode,
    )

    # Metrics
    self.reset_metrics()

    # Create spaces for adversary agent's specs.
    self.adversary_action_dim = (size - 2)**2
    self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
    self.adversary_ts_obs_space = gym.spaces.Box(
        low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
    self.adversary_randomz_obs_space = gym.spaces.Box(
        low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
    self.adversary_image_obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.width, self.height, 3),
        dtype='uint8')

    # Adversary observations are dictionaries containing an encoding of the
    # grid, the current time step, and a randomly generated vector used to
    # condition generation (as in a GAN).
    self.adversary_observation_space = gym.spaces.Dict(
        {'image': self.adversary_image_obs_space,
         'time_step': self.adversary_ts_obs_space,
         'random_z': self.adversary_randomz_obs_space})

    # NetworkX graph used for computing shortest path
    self.graph = grid_graph(dim=[size-2, size-2])
    self.wall_locs = []

  @property
  def processed_action_dim(self):
    return 1

  def _gen_grid(self, width, height):
    """Grid is initially empty, because adversary will create it."""
    # Create an empty grid
    self.grid = multigrid.Grid(width, height)

    # Generate the surrounding walls
    self.grid.wall_rect(0, 0, width, height)

  def get_goal_x(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[0]

  def get_goal_y(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[1]

  def reset_metrics(self):
    self.distance_to_goal = -1
    self.n_clutter_placed = 0
    self.deliberate_agent_placement = -1
    self.passable = -1
    # self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1
    self.shortest_path_length = 0

  def reset(self, *, seed=None, options=None):
    """Fully resets the environment to an empty grid with no agent or goal."""
    if self.fixed_environment:
      seed = self.seed_value
    gym.Env.reset(self, seed=seed)

    self.graph = grid_graph(dim=[self.width-2, self.height-2])
    self.wall_locs = []

    self.step_count = 0
    self.adversary_step_count = 0

    self.agent_start_dir = self._rand_int(0, 4)

    # Current position and direction of the agent
    self.reset_agent_status()

    self.agent_start_pos = None
    self.goal_pos = None

    # Extra metrics
    self.reset_metrics()

    # Generate the grid. Will be random by default, or same environment if
    # 'fixed_environment' is True.
    self._gen_grid(self.width, self.height)

    image = self.grid.encode()
    obs = {
        'image': image,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z()
    }

    return obs, {}

  def reset_agent_status(self):
    """Reset the agent's position, direction, done, and carrying status."""
    self.agent_pos = [None] * self.n_agents
    self.agent_dir = [self.agent_start_dir] * self.n_agents
    self.done = [False] * self.n_agents
    self.carrying = [None] * self.n_agents

  def reset_agent(self):
    """Resets the agent's start position, but leaves goal and walls."""
    # Remove the previous agents from the world
    for a in range(self.n_agents):
      if self.agent_pos[a] is not None:
        self.grid.set(self.agent_pos[a][0], self.agent_pos[a][1], None)

    # Current position and direction of the agent
    self.reset_agent_status()

    if self.agent_start_pos is None:
      raise ValueError('Trying to place agent at empty start position.')
    else:
      self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)

    for a in range(self.n_agents):
      assert self.agent_pos[a] is not None
      assert self.agent_dir[a] is not None

      # Check that the agent doesn't overlap with an object
      start_cell = self.grid.get(*self.agent_pos[a])
      if not (start_cell is None or start_cell.type == 'agent' or
              start_cell.can_overlap()):
        raise ValueError('Wrong object in agent start position.')

    # Step count since episode start
    self.step_count = 0

    # Return first observation
    obs = self.gen_obs()

    return obs, {}

  def reset_to_level(self, level):
    self.reset()
    actions = [int(a) for a in level.split()]
    for a in actions:
      obs, _, done, _, _ = self.step_adversary(a)
      if done:
        return self.reset_agent()

  def remove_wall(self, x, y):
    if (x-1, y-1) in self.wall_locs:
      self.wall_locs.remove((x-1, y-1))
    obj = self.grid.get(x, y)
    if obj is not None and obj.type == 'wall':
      self.grid.set(x, y, None)
      # Keep the metric in sync when the goal or agent overwrites a wall
      self.n_clutter_placed = max(self.n_clutter_placed - 1, 0)

  def compute_shortest_path(self):
    if self.agent_start_pos is None or self.goal_pos is None:
      return

    self.distance_to_goal = abs(
        self.goal_pos[0] - self.agent_start_pos[0]) + abs(
            self.goal_pos[1] - self.agent_start_pos[1])

    # Check if there is a path between agent start position and goal. Remember
    # to subtract 1 due to outside walls existing in the Grid, but not in the
    # networkx graph.
    self.passable = nx.has_path(
        self.graph,
        source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
        target=(self.goal_pos[0]-1, self.goal_pos[1]-1))
    if self.passable:
      # Compute shortest path
      self.shortest_path_length = nx.shortest_path_length(
          self.graph,
          source=(self.agent_start_pos[0]-1, self.agent_start_pos[1]-1),
          target=(self.goal_pos[0]-1, self.goal_pos[1]-1))
    else:
      # Impassable environments have a shortest path length 1 longer than
      # longest possible path
      # self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1
      self.shortest_path_length = 0

  def generate_random_z(self):
    return self.np_random.uniform(size=(self.random_z_dim,)).astype(np.float32)

  def step_adversary(self, loc):
    """The adversary gets n_clutter + 2 moves to place the goal, agent, blocks.

    The action space is the number of possible squares in the grid. The squares
    are numbered from left to right, top to bottom.

    Args:
      loc: An integer specifying the location to place the next object which
        must be decoded into x, y coordinates.

    Returns:
      Standard gymnasium step tuple: observation, reward (always 0),
      terminated, truncated (always False), and info
    """
    loc = int(loc)
    if loc >= self.adversary_action_dim:
      raise ValueError('Position passed to step_adversary is outside the grid.')

    # Add offset of 1 for outside walls
    x = loc % (self.width - 2) + 1
    y = loc // (self.width - 2) + 1
    done = False

    if self.choose_goal_last:
      should_choose_goal = self.adversary_step_count == self.adversary_max_steps - 2
      should_choose_agent = self.adversary_step_count == self.adversary_max_steps - 1
    else:
      should_choose_goal = self.adversary_step_count == 0
      should_choose_agent = self.adversary_step_count == 1

    # Place goal
    if should_choose_goal:
      # If there is goal noise, sometimes randomly place the goal
      if self.np_random.random() < self.goal_noise:
        self.goal_pos = self.place_obj(Goal(), max_tries=100)
      else:
        self.remove_wall(x, y)  # Remove any walls that might be in this loc
        self.put_obj(Goal(), x, y)
        self.goal_pos = (x, y)

    # Place the agent
    elif should_choose_agent:
      self.remove_wall(x, y)  # Remove any walls that might be in this loc

      # Goal has already been placed here
      if self.grid.get(x, y) is not None:
        # Place agent randomly
        self.agent_start_pos = self.place_one_agent(0, rand_dir=False)
        self.deliberate_agent_placement = 0
      else:
        self.agent_start_pos = np.array([x, y])
        self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)
        self.deliberate_agent_placement = 1

    # Place wall
    elif self.adversary_step_count < self.adversary_max_steps:
      # If there is already an object there, action does nothing
      if self.grid.get(x, y) is None:
        self.put_obj(Wall(), x, y)
        self.n_clutter_placed += 1
        self.wall_locs.append((x-1, y-1))

    self.adversary_step_count += 1

    # End of episode
    if self.adversary_step_count >= self.adversary_max_steps:
      done = True
      # Build graph after we are certain agent and goal are placed
      for w in self.wall_locs:
        self.graph.remove_node(w)
      self.compute_shortest_path()

    image = self.grid.encode()
    obs = {
        'image': image,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z()
    }

    return obs, 0, done, False, {}

  def reset_random(self):
    """
    Note, this is based on the original PAIRED implementation from
    https://github.com/google-research/google-research/blob/master/social_rl/gym_multigrid/envs/adversarial.py,
    which sets the domain randomization baseline to use n_clutter/2 blocks.
    """
    if self.fixed_environment:
      if len(self.wall_locs) > 0:
        return self.reset_agent()

    self.reset()
    tmp_adversary_max_steps = self.adversary_max_steps
    self.adversary_max_steps = round(self.n_clutter/2) + 2
    for _ in range(round(self.n_clutter/2) + 2):
      action = self._rand_int(0, self.adversary_action_dim)
      self.step_adversary(action)

    self.compute_shortest_path()
    self.n_clutter_placed = len(self.wall_locs)

    self.adversary_max_steps = tmp_adversary_max_steps

    return self.reset_agent()


class ReparameterizedAdversarialEnv(AdversarialEnv):
  """Grid world where an adversary builds the environment the agent plays.

  In this version, the adversary takes an action for each square in the grid.
  There is no limit on the number of blocks it can place. The action space has
  dimension 4; at each step the adversary can place the goal, agent, a wall, or
  nothing. If it chooses to place the goal or agent when they have previously
  been placed at a different location, they will move to the new location.
  """

  def __init__(self, n_clutter=50, size=15, agent_view_size=5, max_steps=250,
               **kwargs):
    super().__init__(n_clutter=n_clutter, size=size,
                     agent_view_size=agent_view_size, max_steps=max_steps,
                     **kwargs)

    # Adversary has four actions: place agent, goal, wall, or nothing
    self.adversary_action_dim = 4
    self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)

    # Reparam adversaries have additional inputs for the current x,y coords
    self.adversary_xy_obs_space = gym.spaces.Box(
        low=1, high=size-2, shape=(1,), dtype='uint8')

    # Observations are dictionaries containing an encoding of the grid and the
    # agent's direction
    self.adversary_observation_space = gym.spaces.Dict(
        {'image': self.adversary_image_obs_space,
         'time_step': self.adversary_ts_obs_space,
         'random_z': self.adversary_randomz_obs_space,
         'x': self.adversary_xy_obs_space,
         'y': self.adversary_xy_obs_space})

    self.adversary_max_steps = (size - 2)**2

    self.wall_locs = []

  def reset(self, *, seed=None, options=None):
    self.wall_locs = []
    obs, info = super().reset(seed=seed, options=options)
    obs['x'] = [1]
    obs['y'] = [1]
    return obs, info

  def select_random_grid_position(self, exclude=()):
    """Pick a random interior cell, avoiding any positions in `exclude`."""
    exclude = {tuple(int(v) for v in p) for p in exclude if p is not None}
    while True:
      pos = np.array([
          self._rand_int(1, self.grid.width-1),
          self._rand_int(1, self.grid.height-1)
      ])
      if tuple(int(v) for v in pos) not in exclude:
        return pos

  def get_xy_from_step(self, step):
    # Add offset of 1 for outside walls
    x = step % (self.width - 2) + 1
    y = step // (self.width - 2) + 1
    return x, y

  def step_adversary(self, action):
    """The adversary gets a step for each available square in the grid.

    At each step it chooses whether to place the goal, the agent, a block, or
    nothing. If it chooses agent or goal and they have already been placed, they
    will be moved to the new location.

    Args:
      action: An integer in range 0-3 specifying which object to place:
        0 = goal
        1 = agent
        2 = wall
        3 = nothing

    Returns:
      Standard gymnasium step tuple: observation, reward (always 0),
      terminated, truncated (always False), and info
    """
    done = False

    if self.adversary_step_count < self.adversary_max_steps:
      x, y = self.get_xy_from_step(self.adversary_step_count)

      # Place goal
      if action == 0:
        if self.goal_pos is None:
          self.put_obj(Goal(), x, y)
        else:
          goal = self.grid.get(self.goal_pos[0], self.goal_pos[1])
          self.grid.set(self.goal_pos[0], self.goal_pos[1], None)
          self.put_obj(goal, x, y)
        self.goal_pos = (x, y)

      # Place the agent
      elif action == 1:
        if self.agent_start_pos is not None:
          agent = self.grid.get(
              self.agent_start_pos[0], self.agent_start_pos[1])
          self.grid.set(self.agent_start_pos[0], self.agent_start_pos[1], None)
        else:
          agent = None
        self.agent_start_pos = np.array([x, y])
        self.place_agent_at_pos(
            0, self.agent_start_pos, rand_dir=False, agent_obj=agent)

      # Place wall
      elif action == 2:
        self.put_obj(Wall(), x, y)
        self.n_clutter_placed += 1

        self.wall_locs.append((x-1, y-1))

    self.adversary_step_count += 1

    # End of episode
    if self.adversary_step_count >= self.adversary_max_steps:
      done = True

      # If the adversary has not placed the agent or goal, place them randomly
      # (never on top of each other)
      if self.agent_start_pos is None:
        self.agent_start_pos = self.select_random_grid_position(
            exclude=[self.goal_pos])
        # If wall exists here, remove it
        self.remove_wall(self.agent_start_pos[0], self.agent_start_pos[1])
        self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)
        self.deliberate_agent_placement = 0
      else:
        self.deliberate_agent_placement = 1

      if self.goal_pos is None:
        self.goal_pos = self.select_random_grid_position(
            exclude=[self.agent_start_pos])
        # If wall exists here, remove it
        self.remove_wall(self.goal_pos[0], self.goal_pos[1])
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

      # Build graph after we are certain agent and goal are placed
      for w in self.wall_locs:
        self.graph.remove_node(w)
      self.compute_shortest_path()
    else:
      x, y = self.get_xy_from_step(self.adversary_step_count)

    image = self.grid.encode()
    obs = {
        'image': image,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z(),
        'x': [x],
        'y': [y]
    }

    return obs, 0, done, False, {}


class MiniAdversarialEnv(AdversarialEnv):
  def __init__(self, **kwargs):
    super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50,
                     **kwargs)


class MiniReparameterizedAdversarialEnv(ReparameterizedAdversarialEnv):
  def __init__(self, **kwargs):
    super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50,
                     **kwargs)


class NoisyAdversarialEnv(AdversarialEnv):
  def __init__(self, **kwargs):
    super().__init__(goal_noise=0.3, **kwargs)


class MediumAdversarialEnv(AdversarialEnv):
  def __init__(self, **kwargs):
    super().__init__(n_clutter=30, size=10, agent_view_size=5, max_steps=200,
                     **kwargs)


class GoalLastAdversarialEnv(AdversarialEnv):
  def __init__(self, fixed_environment=False, seed=None, **kwargs):
    super().__init__(choose_goal_last=True, fixed_environment=fixed_environment,
                     seed=seed, max_steps=250, **kwargs)


class GoalLastOpaqueWallsAdversarialEnv(AdversarialEnv):
  def __init__(self, fixed_environment=False, seed=None, **kwargs):
    super().__init__(
      choose_goal_last=True, see_through_walls=False,
      fixed_environment=fixed_environment, seed=seed, max_steps=250, **kwargs)


class GoalLastFewerBlocksAdversarialEnv(AdversarialEnv):
  def __init__(self, fixed_environment=False, seed=None, **kwargs):
    super().__init__(
      choose_goal_last=True, n_clutter=25,
      fixed_environment=fixed_environment, seed=seed, max_steps=250, **kwargs)


class GoalLastFewerBlocksOpaqueWallsAdversarialEnv(AdversarialEnv):
  def __init__(self, fixed_environment=False, seed=None, **kwargs):
    super().__init__(
      choose_goal_last=True, n_clutter=25, see_through_walls=False,
      fixed_environment=fixed_environment, seed=seed, max_steps=250, **kwargs)


class MiniGoalLastAdversarialEnv(AdversarialEnv):
  def __init__(self, fixed_environment=False, seed=None, **kwargs):
    super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50,
                     choose_goal_last=True, fixed_environment=fixed_environment,
                     seed=seed, **kwargs)


class FixedAdversarialEnv(AdversarialEnv):
  def __init__(self, **kwargs):
    super().__init__(n_clutter=50, size=15, agent_view_size=5, max_steps=50,
                     fixed_environment=True, **kwargs)


class EmptyMiniFixedAdversarialEnv(AdversarialEnv):
  def __init__(self, **kwargs):
    super().__init__(n_clutter=0, size=6, agent_view_size=5, max_steps=50,
                     fixed_environment=True, **kwargs)


# reset() returns the adversary's observation rather than one from
# observation_space, so gymnasium's passive env checker is disabled.
_REGISTRATIONS = [
    ('MultiGrid-Adversarial-v0', 'AdversarialEnv', 250),
    ('MultiGrid-ReparameterizedAdversarial-v0',
     'ReparameterizedAdversarialEnv', 250),
    ('MultiGrid-MiniAdversarial-v0', 'MiniAdversarialEnv', 50),
    ('MultiGrid-MiniReparameterizedAdversarial-v0',
     'MiniReparameterizedAdversarialEnv', 50),
    ('MultiGrid-NoisyAdversarial-v0', 'NoisyAdversarialEnv', 250),
    ('MultiGrid-MediumAdversarial-v0', 'MediumAdversarialEnv', 200),
    ('MultiGrid-GoalLastAdversarial-v0', 'GoalLastAdversarialEnv', 250),
    ('MultiGrid-GoalLastOpaqueWallsAdversarial-v0',
     'GoalLastOpaqueWallsAdversarialEnv', 250),
    ('MultiGrid-GoalLastFewerBlocksAdversarial-v0',
     'GoalLastFewerBlocksAdversarialEnv', 250),
    ('MultiGrid-GoalLastFewerBlocksOpaqueWallsAdversarial-v0',
     'GoalLastFewerBlocksOpaqueWallsAdversarialEnv', 250),
    ('MultiGrid-MiniGoalLastAdversarial-v0', 'MiniGoalLastAdversarialEnv', 50),
    ('MultiGrid-FixedAdversarial-v0', 'FixedAdversarialEnv', 50),
    ('MultiGrid-EmptyMiniFixedAdversarial-v0',
     'EmptyMiniFixedAdversarialEnv', 50),
]

for env_id, class_name, max_episode_steps in _REGISTRATIONS:
  if env_id not in gym.registry:
    gym.register(
        id=env_id,
        entry_point=f'{__name__}:{class_name}',
        max_episode_steps=max_episode_steps,
        disable_env_checker=True,
    )
