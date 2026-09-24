"""Level helpers, plus an ASCII-grid parser for a text-generation baseline.

build_level, level_metrics and env_to_rows are used by every run. The parser
(parse_layout, layout_to_actions) is for a future baseline where a model
writes the whole level as text; jev's structured reads don't use it.

In that baseline the model is asked to draw the interior of the level (outer
walls excluded) as an ASCII grid, one row per line:

  '#' wall    '.' empty    'A' agent start    'G' goal

parse_layout() is deliberately lenient: it pulls the grid out of surrounding
prose or code fences, accepts common symbol aliases, strips an outer border if
the model drew one, and pads/truncates to the right size. Every repair is
recorded in Layout.issues so format adherence can be scored separately from
level quality.

layout_to_actions() then encodes the layout for either adversary:
  * ReparameterizedAdversarialEnv: one action per cell (lossless).
  * AdversarialEnv: goal/agent/wall locations, capped at n_clutter walls.
"""
import dataclasses
import re

import numpy as np

from .envs.adversarial import AdversarialEnv, ReparameterizedAdversarialEnv

WALL, EMPTY, AGENT, GOAL = '#', '.', 'A', 'G'

# Common deviations models make, mapped onto the canonical symbols.
CHAR_ALIASES = {
    'W': WALL, 'X': WALL, 'x': WALL, '1': WALL, '█': WALL, '■': WALL,
    '-': EMPTY, '_': EMPTY, '0': EMPTY, 'o': EMPTY, '·': EMPTY, '□': EMPTY,
    'a': AGENT, 'S': AGENT, 's': AGENT, '@': AGENT,
    'g': GOAL, 'E': GOAL, 'e': GOAL, '*': GOAL,
}
_CANONICAL = {WALL, EMPTY, AGENT, GOAL}
_FENCE_RE = re.compile(r'```[^\n]*\n(.*?)```', re.DOTALL)

# Reparameterized adversary actions (see ReparameterizedAdversarialEnv).
_REPARAM_ACTION = {GOAL: 0, AGENT: 1, WALL: 2, EMPTY: 3}


@dataclasses.dataclass
class Layout:
  """An inner_size x inner_size level, in interior coordinates (0-indexed)."""
  rows: list[str]
  grid_found: bool
  issues: list[str] = dataclasses.field(default_factory=list)

  @property
  def size(self):
    return len(self.rows)

  def _find(self, symbol):
    for y, row in enumerate(self.rows):
      x = row.find(symbol)
      if x >= 0:
        return (x, y)
    return None

  @property
  def agent_pos(self):
    return self._find(AGENT)

  @property
  def goal_pos(self):
    return self._find(GOAL)

  @property
  def walls(self):
    return [(x, y) for y, row in enumerate(self.rows)
            for x, c in enumerate(row) if c == WALL]

  def to_text(self):
    return '\n'.join(self.rows)


def _normalize_line(line):
  """Returns the line as canonical symbols, or None if it isn't a grid row."""
  chars = ''.join(line.split())  # Tolerate spaces between cells
  if not chars:
    return None
  out = []
  for c in chars:
    c = CHAR_ALIASES.get(c, c)
    if c not in _CANONICAL:
      return None
    out.append(c)
  return ''.join(out)


def _grid_runs(text):
  """Yields maximal runs of consecutive grid-like lines in the text."""
  run = []
  for line in text.splitlines():
    row = _normalize_line(line)
    if row is not None and len(row) >= 3:
      run.append(row)
    else:
      if run:
        yield run
      run = []
  if run:
    yield run


def _extract_rows(text):
  """Finds the most plausible grid; prefers the last fenced block."""
  for block in reversed(_FENCE_RE.findall(text)):
    runs = list(_grid_runs(block))
    if runs:
      return max(runs, key=len)
  runs = list(_grid_runs(text))
  if not runs:
    return None
  # Longest run wins; on ties prefer the last (models often revise).
  return max(reversed(runs), key=len)


def _strip_border(rows, inner_size):
  """Removes an all-wall outer border if the model drew the full grid."""
  if len(rows) != inner_size + 2:
    return rows, False
  if not all(set(r) == {WALL} for r in (rows[0], rows[-1])):
    return rows, False
  if not all(len(r) >= 2 and r[0] == WALL and r[-1] == WALL for r in rows):
    return rows, False
  return [r[1:-1] for r in rows[1:-1]], True


def _keep_first(rows, symbol, issues):
  """Keeps only the first occurrence of a unique symbol (agent or goal)."""
  seen = False
  out = []
  for row in rows:
    chars = list(row)
    for i, c in enumerate(chars):
      if c == symbol:
        if seen:
          chars[i] = EMPTY
        seen = True
    out.append(''.join(chars))
  count = sum(r.count(symbol) for r in rows)
  if count > 1:
    issues.append(f'{count} "{symbol}" cells; kept the first')
  elif count == 0:
    issues.append(f'no "{symbol}" cell')
  return out


def parse_layout(text, inner_size):
  """Parses model output into an inner_size x inner_size Layout."""
  rows = _extract_rows(text)
  if rows is None:
    return Layout(rows=[EMPTY * inner_size] * inner_size, grid_found=False,
                  issues=['no grid found in output'])

  issues = []
  rows, stripped = _strip_border(rows, inner_size)
  if stripped:
    issues.append('stripped outer border')

  if len(rows) != inner_size:
    issues.append(f'{len(rows)} rows instead of {inner_size}')
  widths = sorted({len(r) for r in rows})
  if widths != [inner_size]:
    issues.append(f'row widths {widths} instead of {inner_size}')
  rows = [r[:inner_size].ljust(inner_size, EMPTY) for r in rows[:inner_size]]
  rows += [EMPTY * inner_size] * (inner_size - len(rows))

  rows = _keep_first(rows, AGENT, issues)
  rows = _keep_first(rows, GOAL, issues)
  return Layout(rows=rows, grid_found=True, issues=issues)


def _random_free_cell(layout, rng, exclude):
  free = [(x, y) for y, row in enumerate(layout.rows)
          for x, c in enumerate(row) if c == EMPTY and (x, y) not in exclude]
  if not free:  # Fully walled level: fall back to any cell
    free = [(x, y) for y in range(layout.size) for x in range(layout.size)
            if (x, y) not in exclude]
  return free[rng.integers(len(free))]


def layout_to_actions(layout, env, rng=None):
  """Encodes a Layout as the sequence of step_adversary() actions for env.

  Args:
    layout: Parsed Layout whose size matches env's interior.
    env: An unwrapped AdversarialEnv or ReparameterizedAdversarialEnv.
    rng: numpy Generator used to fill in a missing agent/goal for the
      location-based adversary (the reparameterized env does this itself).

  Returns:
    (actions, issues): list of int actions, and any encoding caveats.
  """
  inner = env.width - 2
  assert layout.size == inner, (layout.size, inner)

  if isinstance(env, ReparameterizedAdversarialEnv):
    return [_REPARAM_ACTION[c] for row in layout.rows for c in row], []

  if not isinstance(env, AdversarialEnv):
    raise TypeError(f'Unsupported env type {type(env).__name__}')

  rng = rng if rng is not None else np.random.default_rng()
  issues = []
  goal = layout.goal_pos
  if goal is None:
    goal = _random_free_cell(layout, rng, exclude=())
    issues.append('goal placed randomly')
  agent = layout.agent_pos
  if agent is None:
    agent = _random_free_cell(layout, rng, exclude={goal})
    issues.append('agent placed randomly')

  walls = layout.walls
  if len(walls) > env.n_clutter:
    issues.append(f'{len(walls)} walls exceed n_clutter={env.n_clutter}; '
                  'kept the first in row-major order')
    walls = walls[:env.n_clutter]

  to_loc = lambda p: p[1] * inner + p[0]
  # Extra wall steps target the goal cell: a no-op when the goal is already
  # there, and overwritten by the goal otherwise (choose_goal_last).
  wall_actions = [to_loc(w) for w in walls]
  wall_actions += [to_loc(goal)] * (env.n_clutter - len(walls))
  if env.choose_goal_last:
    actions = wall_actions + [to_loc(goal), to_loc(agent)]
  else:
    actions = [to_loc(goal), to_loc(agent)] + wall_actions
  return actions, issues


def build_level(env, actions, seed=None):
  """Resets env and plays the adversary actions until the level is built."""
  env.reset(seed=seed)
  for action in actions:
    _, _, done, _, _ = env.step_adversary(action)
    if done:
      break
  else:
    raise RuntimeError('Adversary actions ended before the level was built.')
  env.reset_agent()


def level_metrics(env):
  """Structural metrics of the level currently built in env."""
  return {
      'passable': bool(env.passable),
      'shortest_path_length': int(env.shortest_path_length),
      'distance_to_goal': int(env.distance_to_goal),
      'n_walls': len(env.wall_locs),
      'wall_density': len(env.wall_locs) / (env.width - 2) ** 2,
      'deliberate_agent_placement': int(env.deliberate_agent_placement),
      'agent_pos': [int(v) - 1 for v in env.agent_start_pos],
      'goal_pos': [int(v) - 1 for v in env.goal_pos],
  }


def env_to_rows(env, symbols=None):
  """Reads the built level back out of env as interior ASCII rows.

  symbols optionally overrides the characters for 'empty', 'wall', 'agent' and
  'goal'.
  """
  symbols = {**{'empty': EMPTY, 'wall': WALL, 'agent': AGENT, 'goal': GOAL},
             **(symbols or {})}
  rows = []
  for y in range(1, env.height - 1):
    row = []
    for x in range(1, env.width - 1):
      cell = env.grid.get(x, y)
      row.append(symbols[cell.type if cell else 'empty'])
    rows.append(''.join(row))
  return rows
