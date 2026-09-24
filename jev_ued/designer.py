"""Middle component: jev designs a level row by row as the UED adversary.

For each row, top to bottom, jev gets one `choice` question per cell (empty,
wall, agent or goal) with the description and the rows built so far as the
state. Its per-cell probabilities are the adversary's action distribution:
an action is drawn per cell and played into ReparameterizedAdversarialEnv,
which walks the grid in the same row-major order.

The level keeps exactly one agent and one goal: once placed, that option is
no longer offered, and if several cells in one row pick it, the most probable
cell keeps it and the others are redrawn without it. If jev never places one,
the env places it at random (deliberate_agent_placement = 0).
"""
import dataclasses
import math

import numpy as np

from .envs.adversarial import ReparameterizedAdversarialEnv

EMPTY, WALL, AGENT, GOAL = 'empty', 'wall', 'agent', 'goal'
UNIQUE = (AGENT, GOAL)

# Cell options offered to jev (option name -> description).
OPTIONS = {
    EMPTY: 'open floor the agent can walk on',
    WALL: 'a wall block the agent cannot pass through',
    AGENT: "the agent's start cell; a level has exactly one",
    GOAL: 'the goal the agent must reach; a level has exactly one',
}

# ReparameterizedAdversarialEnv.step_adversary actions.
ACTION = {GOAL: 0, AGENT: 1, WALL: 2, EMPTY: 3}
# Agent is 'S' (start): the server labels answer options A, B, C, ...
SYMBOL = {EMPTY: '.', WALL: '#', AGENT: 'S', GOAL: 'G'}

INSTRUCTIONS = """\
You are designing a {n}x{n} grid-world level that matches the description in \
the state. An agent moves up, down, left and right between open cells, and \
cannot pass through walls. A boundary wall surrounds the grid and is not part \
of it.

The level is built one row at a time, top to bottom. The state gives the rows \
built so far ('.' empty, '#' wall, 'S' agent start, 'G' goal) and which row is being \
built now. Each question asks what goes in one column of the current row."""


def _cell_id(x):
  # Digits tokenize one per token in Gemma, so the answer label after the id
  # stays a separate token, as the server requires.
  return f'x{x + 1}'


@dataclasses.dataclass
class Design:
  """A level jev designed, with everything needed to analyze it."""
  rows: list[str]  # '.#AG' rows as chosen by jev
  actions: list[int]  # step_adversary actions, row-major
  probabilities: list[list[dict[str, float]]]  # [row][col] -> option -> p
  log_likelihood: float  # sum of log p(chosen) under jev's distributions
  mean_entropy: float  # mean per-cell entropy (nats)
  requests: int
  reads: int  # denoising reads across all requests
  chunks_per_row: list[int]  # >1 means a row didn't fit one canvas read


def _entropy(p):
  return -sum(v * math.log(v) for v in p.values() if v > 0)


def _pick(probs, rng, decode, exclude=()):
  """Draws (or argmaxes) an option, renormalized without `exclude`."""
  names = [k for k in probs if k not in exclude]
  weights = np.array([probs[k] for k in names], dtype=float)
  if weights.sum() <= 0:
    weights = np.ones(len(names))
  if decode == 'argmax':
    return names[int(weights.argmax())]
  return names[rng.choice(len(names), p=weights / weights.sum())]


def _resolve_unique(choices, probs, rng, decode, placed):
  """Keeps at most one cell per unique option; redraws the rest."""
  for option in UNIQUE:
    cells = [x for x, c in enumerate(choices) if c == option]
    if len(cells) <= 1:
      continue
    keep = max(cells, key=lambda x: probs[x][option])
    for x in cells:
      if x != keep:
        choices[x] = _pick(probs[x], rng, decode, exclude=UNIQUE)
  placed.update(c for c in choices if c in UNIQUE)
  return choices


def row_request(description, rows_so_far, row, n, placed):
  """State and questions asking jev for one row."""
  offered = {k: v for k, v in OPTIONS.items() if k not in placed}
  state = {
      'description': description,
      'grid_size': f'{n}x{n}',
      'rows_built': rows_so_far or ['(none yet)'],
      'current_row': f'{row + 1} of {n}',
      'agent_placed': AGENT in placed,
      'goal_placed': GOAL in placed,
  }
  questions = {
      _cell_id(x): {
          'type': 'choice',
          # No row number here: the system prompt stays identical across
          # rows, so vLLM's prefix cache can reuse it.
          'instructions': f'What goes in column {x + 1} of the current row?',
          'criteria': offered,
      }
      for x in range(n)
  }
  return state, questions


def design_level(client, env, description, rng, decode='sample', seed=0,
                 **extensions):
  """Has jev design a level in env, one row per request.

  Args:
    client: JevClient.
    env: Unwrapped ReparameterizedAdversarialEnv; it is reset here and holds
      the built level afterwards (agent reset and ready to play).
    description: Natural-language level description.
    rng: numpy Generator for drawing actions from jev's distributions.
    decode: 'sample' draws from the distributions; 'argmax' takes the mode.
    seed: Base seed for the server's noise draws.
    **extensions: Passed to JevClient.decide (samples, think, steps, ...).

  Returns:
    Design.
  """
  if not isinstance(env, ReparameterizedAdversarialEnv):
    raise TypeError('design_level needs a ReparameterizedAdversarialEnv')
  n = env.width - 2
  env.reset(seed=seed)
  instructions = INSTRUCTIONS.format(n=n)

  rows, actions, all_probs, placed = [], [], [], set()
  log_likelihood, entropies, reads, chunks = 0.0, [], 0, []
  for row in range(n):
    state, questions = row_request(description, rows, row, n, placed)
    body = client.decide(state, questions, seed=seed * 1000 + row,
                         instructions=instructions, **extensions)
    diagnostics = body.get('diagnostics', {})
    reads += diagnostics.get('timing', {}).get('reads', 0)
    chunks.append(len(diagnostics.get('chunks') or [None]))

    probs = [body['answers'][_cell_id(x)]['probabilities'] for x in range(n)]
    choices = [_pick(p, rng, decode) for p in probs]
    choices = _resolve_unique(choices, probs, rng, decode, placed)

    for p, c in zip(probs, choices):
      log_likelihood += math.log(max(p.get(c, 0.0), 1e-12))
      entropies.append(_entropy(p))
      actions.append(ACTION[c])
      env.step_adversary(ACTION[c])
    rows.append(''.join(SYMBOL[c] for c in choices))
    all_probs.append(probs)

  env.reset_agent()
  return Design(rows=rows, actions=actions, probabilities=all_probs,
                log_likelihood=log_likelihood,
                mean_entropy=float(np.mean(entropies)),
                requests=n, reads=reads, chunks_per_row=chunks)
