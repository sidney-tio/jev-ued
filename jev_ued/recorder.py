"""Step-level data for jev-artist runs, and rendering recorded states to PNG.

Each drawing turn becomes one JSON line in steps.jsonl:
  topic_id, repeat, prompt  which target image, and which of its k drawings
  turn                      the time step (0-indexed)
  state                     exactly what jev was shown (subject, canvas, turn)
  probabilities             row -> {column id or "-1": p} for every asked row
  argmax                    row -> jev's most likely option
  chosen                    row -> option actually applied (differs from
                            argmax only with --decode sample)
  canvas_after              the canvas once the turn's pixels are filled
  no_op_probability, entropy, reads, seed

Render recorded states:
  python -m jev_ued.recorder runs/artist-.../steps.jsonl
  python -m jev_ued.recorder steps.jsonl --topic smiley --repeat 0 --turn 3 --after
"""
import argparse
import json
import math
import pathlib
import threading

from PIL import Image

from .render import render_maze


def _entropy(p):
  return -sum(v * math.log(v) for v in p.values() if v > 0)


def step_record(topic_id, repeat, prompt, turn_index, turn, seed=None):
  """Flattens an artist Turn into a JSON-serializable step record."""
  rows = sorted(turn.probabilities)
  return {
      'topic_id': topic_id,
      'repeat': repeat,
      'prompt': prompt,
      'turn': turn_index,
      'seed': seed,
      'state': turn.state,
      'probabilities': {str(y): turn.probabilities[y] for y in rows},
      'argmax': {str(y): turn.argmax[y] for y in rows},
      'chosen': {str(y): turn.chosen[y] for y in rows},
      'canvas_after': [''.join(map(str, r)) for r in turn.canvas_after],
      'no_op_probability': {str(y): turn.probabilities[y].get('-1')
                            for y in rows},
      'entropy': {str(y): _entropy(turn.probabilities[y]) for y in rows},
      'reads': turn.reads,
  }


class DataWriter:
  """Appends step records to a JSONL file, flushing after every step.

  Safe to share across threads.
  """

  def __init__(self, path):
    self.path = pathlib.Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._file = open(self.path, 'a')
    self._lock = threading.Lock()

  def write(self, record):
    line = json.dumps(record) + '\n'
    with self._lock:
      self._file.write(line)
      self._file.flush()

  def close(self):
    self._file.close()

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()


def load_steps(path, topic_id=None, repeat=None, turn=None):
  """Reads step records, optionally filtered by topic, repeat and/or turn.

  Records come back ordered by topic, repeat and turn (parallel drawings
  interleave in the file).
  """
  records = []
  with open(path) as f:
    for line in f:
      r = json.loads(line)
      if topic_id is not None and r['topic_id'] != topic_id:
        continue
      if repeat is not None and r['repeat'] != repeat:
        continue
      if turn is not None and r['turn'] != turn:
        continue
      records.append(r)
  return sorted(records, key=lambda r: (r['topic_id'], r['repeat'], r['turn']))


def canvas_of(record, after=False):
  """The record's canvas as [row][col] ints: the state jev saw, or the
  canvas after the turn's fills."""
  rows = record['canvas_after'] if after else record['state']['canvas']
  return [[int(c) for c in row] for row in rows]


def state_to_png(record, path, after=False, tile_size=32):
  """Renders a recorded state as a maze PNG (1 = wall)."""
  image = render_maze(canvas_of(record, after), tile_size=tile_size)
  Image.fromarray(image).save(path)
  return path


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument('steps', help='steps.jsonl written by jev_ued.artist')
  p.add_argument('--topic', default=None, help='Topic id, e.g. smiley')
  p.add_argument('--repeat', type=int, default=None)
  p.add_argument('--turn', type=int, default=None)
  p.add_argument('--after', action='store_true',
                 help="Render the canvas after the turn's fills, not the "
                      'state jev was shown')
  p.add_argument('--out', default=None,
                 help='Output directory (default: states/ next to steps)')
  p.add_argument('--tile-size', type=int, default=32)
  args = p.parse_args()

  steps = pathlib.Path(args.steps)
  out_dir = pathlib.Path(args.out) if args.out else steps.parent / 'states'
  out_dir.mkdir(parents=True, exist_ok=True)
  records = load_steps(steps, args.topic, args.repeat, args.turn)
  if not records:
    raise SystemExit('No matching steps.')
  which = 'after' if args.after else 'state'
  for r in records:
    path = (out_dir /
            f"{r['topic_id']}-r{r['repeat']:02d}-t{r['turn']:02d}-{which}.png")
    state_to_png(r, path, after=args.after, tile_size=args.tile_size)
  print(f'Wrote {len(records)} PNGs to {out_dir}')


if __name__ == '__main__':
  main()
