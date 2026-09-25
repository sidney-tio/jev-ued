"""jev-artist: can jev draw? A prompt goes in, a 13x13 binary picture comes out.

The picture is drawn over several turns, one /v1/systemone request per turn.
Each question is a row; its options are the ids of that row's empty columns
(0-12) plus -1, a no-op that leaves the row unchanged. Each turn fills at most
one pixel per row, and a filled column is no longer offered for that row.
Drawing stops when every row answers -1, or after --max-turns.

The finished picture (1 = filled) is rendered as a maze: filled pixels become
walls inside a boundary wall (see render.py; plain numpy, no MiniGrid).

Prompts come from a YAML dataset (data/topics.yaml), and each topic is
drawn --k times with a different seed.

Test run (needs scripts/serve.sh running):
  python -m jev_ued.artist                      # every topic, k=10
  python -m jev_ued.artist --only smiley,heart --k 3

Writes to --out (default runs/artist-<timestamp>/):
  steps.jsonl                one record per topic, repeat and turn: the state
                             jev saw, its per-row probabilities, argmax and
                             chosen options (see recorder.py; render any
                             state with python -m jev_ued.recorder)
  drawings/<topic>-rNN.png   final drawing rendered as a maze
  drawings/<topic>-rNN.txt   final drawing as 0/1 rows
  drawings.jsonl             one summary line per drawing
  config.json                run arguments and the topics used
"""
import argparse
import concurrent.futures
import dataclasses
import datetime
import json
import pathlib
import re

import numpy as np
import yaml
from PIL import Image

from . import recorder
from .client import JevClient
from .render import render_maze

SIZE = 13
NO_OP = '-1'

INSTRUCTIONS = """\
You are drawing a picture of the subject in the state on a {n}x{n} grid of \
pixels. The canvas is given as {n} rows, top to bottom, where 1 is a filled \
pixel and 0 is empty; columns are numbered 0 to {last} from left to right.

The picture is drawn over several turns. Each question is one row: answer \
with the column of the next pixel to fill in that row, or -1 to leave the row \
as it is this turn. Only empty columns are offered. When the picture is \
finished, answer -1 for every row."""


@dataclasses.dataclass
class Turn:
  state: dict  # Exactly what jev was shown this turn
  probabilities: dict[int, dict[str, float]]  # row -> option -> p
  argmax: dict[int, str]  # row -> most likely option
  chosen: dict[int, str]  # row -> option applied (column id or "-1")
  fills: dict[int, int]  # row -> column filled this turn
  canvas_after: list[list[int]]
  reads: int


@dataclasses.dataclass
class Drawing:
  prompt: str
  canvas: list[list[int]]  # [row][col], 1 = filled
  turns: list[Turn]
  finished: bool  # True if jev chose -1 everywhere (vs. hitting max turns)

  def ascii(self):
    return '\n'.join(''.join(map(str, row)) for row in self.canvas)


def turn_request(prompt, canvas, turn, max_turns):
  """State and questions for one turn; full rows are not asked."""
  state = {
      'subject': prompt,
      'canvas': [''.join(map(str, row)) for row in canvas],
      'turn': f'{turn + 1} of at most {max_turns}',
  }
  questions = {}
  for y, row in enumerate(canvas):
    empty = [str(x) for x, v in enumerate(row) if v == 0]
    if not empty:
      continue  # A choice needs at least two options besides the no-op
    questions[f'r{y}'] = {
        'type': 'choice',
        'instructions': f'Row {y}: which column should be filled next?',
        'criteria': {**{x: None for x in empty},
                     NO_OP: 'leave this row unchanged this turn'},
    }
  return state, questions


def _pick(probs, rng, decode):
  names = list(probs)
  weights = np.array([probs[k] for k in names], dtype=float)
  if decode == 'argmax' or weights.sum() <= 0:
    return names[int(weights.argmax())]
  return names[rng.choice(len(names), p=weights / weights.sum())]


def draw(client, prompt, rng, max_turns=SIZE, decode='sample', seed=0,
         size=SIZE, on_turn=None, **extensions):
  """Has jev draw `prompt`, one request per turn.

  Args:
    client: JevClient.
    prompt: What to draw, e.g. "smiley face".
    rng: numpy Generator for sampling from jev's distributions.
    max_turns: Turn limit; each turn fills at most one pixel per row.
    decode: 'sample' draws from each row's distribution; 'argmax' takes the
      mode.
    seed: Base seed for the server's noise draws.
    size: Canvas side length.
    on_turn: Optional callback(turn_index, Drawing) after each turn.
    **extensions: Passed to JevClient.decide (samples, think, steps, ...).

  Returns:
    Drawing.
  """
  canvas = [[0] * size for _ in range(size)]
  drawing = Drawing(prompt=prompt, canvas=canvas, turns=[], finished=False)
  instructions = INSTRUCTIONS.format(n=size, last=size - 1)

  for turn in range(max_turns):
    state, questions = turn_request(prompt, canvas, turn, max_turns)
    if not questions:  # Canvas is full
      drawing.finished = True
      break
    body = client.decide(state, questions, seed=seed * 1000 + turn,
                         instructions=instructions, **extensions)

    probabilities, argmax, chosen, fills = {}, {}, {}, {}
    for qid in questions:
      y = int(qid[1:])
      probs = body['answers'][qid]['probabilities']
      probabilities[y] = probs
      argmax[y] = max(probs, key=probs.get)
      chosen[y] = argmax[y] if decode == 'argmax' else _pick(probs, rng, decode)
      if chosen[y] != NO_OP:
        fills[y] = int(chosen[y])
        canvas[y][fills[y]] = 1
    reads = body.get('diagnostics', {}).get('timing', {}).get('reads', 0)
    drawing.turns.append(Turn(
        state=state, probabilities=probabilities, argmax=argmax,
        chosen=chosen, fills=fills, canvas_after=[r[:] for r in canvas],
        reads=reads))
    if on_turn:
      on_turn(turn, drawing)
    if not fills:  # -1 on every row: jev says it's done
      drawing.finished = True
      break
  return drawing


def load_topics(path, only=None):
  """Reads topics from YAML: a list of {id, prompt, probes?} under 'topics'."""
  with open(path) as f:
    topics = yaml.safe_load(f)['topics']
  ids = [t['id'] for t in topics]
  if len(set(ids)) != len(ids):
    raise ValueError(f'{path}: duplicate topic ids')
  for t in topics:
    if not re.fullmatch(r'[a-z0-9_]+', t['id']) or not t.get('prompt'):
      raise ValueError(f'{path}: bad topic {t!r}; needs a snake_case id and '
                       'a prompt')
  if only:
    unknown = set(only) - set(ids)
    if unknown:
      raise ValueError(f'unknown topic ids: {sorted(unknown)}')
    topics = [t for t in topics if t['id'] in only]
  return topics


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument('--dataset', default='data/topics.yaml',
                 help='YAML file of topics to draw')
  p.add_argument('--only', default=None,
                 help='Comma-separated topic ids to run (default: all)')
  p.add_argument('--k', type=int, default=10,
                 help='Drawings per topic, each with its own seed')
  p.add_argument('--url', default='http://127.0.0.1:8011',
                 help='structured_server.py address')
  p.add_argument('--max-turns', type=int, default=SIZE)
  p.add_argument('--decode', choices=['sample', 'argmax'], default='argmax',
                 help="Take each row's most likely option, or sample it")
  p.add_argument('--samples', default='auto',
                 help='Server noise draws per read: "auto" or a count')
  p.add_argument('--think', type=int, default=0,
                 help='Thought tokens jev may write before each turn')
  p.add_argument('--concurrency', type=int, default=8,
                 help='Drawings in progress at once')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', default=None)
  return p.parse_args()


def main():
  args = parse_args()
  topics = load_topics(args.dataset,
                       args.only.split(',') if args.only else None)
  out_dir = pathlib.Path(args.out or datetime.datetime.now().strftime(
      'runs/artist-%Y%m%d-%H%M%S'))
  (out_dir / 'drawings').mkdir(parents=True, exist_ok=True)
  (out_dir / 'config.json').write_text(json.dumps(
      {**vars(args), 'topics': topics}, indent=2))
  samples = args.samples if args.samples == 'auto' else int(args.samples)

  client = JevClient(args.url)
  print(f'Waiting for {args.url} ...', flush=True)
  client.wait_until_healthy()

  jobs = [(t, rep) for t in topics for rep in range(args.k)]
  print(f'{len(topics)} topics x k={args.k} = {len(jobs)} drawings', flush=True)

  def run(job):
    topic, rep = job
    seed = args.seed * 1_000_000 + topics.index(topic) * 1000 + rep

    def on_turn(turn, drawing):
      writer.write(recorder.step_record(
          topic['id'], rep, drawing.prompt, turn, drawing.turns[-1],
          seed=seed))

    drawing = draw(client, topic['prompt'], np.random.default_rng(seed),
                   max_turns=args.max_turns, decode=args.decode, seed=seed,
                   on_turn=on_turn, samples=samples, think=args.think)
    stem = out_dir / 'drawings' / f"{topic['id']}-r{rep:02d}"
    Image.fromarray(render_maze(drawing.canvas)).save(f'{stem}.png')
    stem.with_suffix('.txt').write_text(drawing.ascii() + '\n')
    return topic, rep, seed, drawing

  done = 0
  with recorder.DataWriter(out_dir / 'steps.jsonl') as writer, \
       open(out_dir / 'drawings.jsonl', 'w') as summary, \
       concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
    futures = [pool.submit(run, job) for job in jobs]
    for future in concurrent.futures.as_completed(futures):
      topic, rep, seed, drawing = future.result()
      pixels = sum(map(sum, drawing.canvas))
      summary.write(json.dumps({
          'topic_id': topic['id'], 'repeat': rep, 'prompt': topic['prompt'],
          'seed': seed, 'turns': len(drawing.turns),
          'finished': drawing.finished, 'pixels': pixels,
          'canvas': drawing.ascii().splitlines(),
      }) + '\n')
      summary.flush()
      done += 1
      print(f"[{done}/{len(jobs)}] {topic['id']} r{rep:02d}: "
            f"{'finished' if drawing.finished else 'hit max turns'} after "
            f'{len(drawing.turns)} turns, {pixels} pixels', flush=True)
  client.close()
  print(f'Steps: {out_dir / "steps.jsonl"}')
  print(f'Drawings: {out_dir / "drawings"}')


if __name__ == '__main__':
  main()
