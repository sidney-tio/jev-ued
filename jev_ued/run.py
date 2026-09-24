"""Runs jev as the UED adversary: descriptions -> row reads -> levels -> metrics.

Needs vLLM and structured_server.py running (see scripts/serve.sh). Example:
  python -m jev_ued.run --n 4

Writes to --out (default runs/<timestamp>/):
  results.jsonl  one record per level: rows, jev's per-cell probabilities,
                 structural metrics
  summary.json   aggregate and per-description metrics
  levels/*.png   rendered levels
"""
import argparse
import collections
import concurrent.futures
import datetime
import json
import pathlib
import statistics

import gymnasium as gym
import numpy as np
from PIL import Image

from . import designer
from . import envs  # noqa: F401  (registers MultiGrid-*-v0)
from . import layout
from .client import JevClient

DESIGNER_SYMBOLS = {designer.AGENT: designer.SYMBOL[designer.AGENT]}


def load_descriptions(path):
  """One description per line; blank lines and '#' comments are skipped."""
  lines = pathlib.Path(path).read_text().splitlines()
  return [l.strip() for l in lines if l.strip() and not l.startswith('#')]


def _mean(values):
  return statistics.fmean(values) if values else None


def summarize(records):
  """Aggregate metrics over a list of result records."""
  ok = [r for r in records if 'error' not in r]
  solvable = [r for r in ok if r['passable']]
  return {
      'n': len(records),
      'errors': len(records) - len(ok),
      'solvable_rate': _mean([r['passable'] for r in ok]),
      'deliberate_agent_rate': _mean(
          [r['deliberate_agent_placement'] == 1 for r in ok]),
      'mean_shortest_path': _mean(
          [r['shortest_path_length'] for r in solvable]),
      'max_shortest_path': max(
          (r['shortest_path_length'] for r in solvable), default=None),
      'mean_wall_density': _mean([r['wall_density'] for r in ok]),
      'mean_cell_entropy': _mean([r['mean_entropy'] for r in ok]),
      'mean_log_likelihood': _mean([r['log_likelihood'] for r in ok]),
      'unique_level_rate': (len({tuple(r['built_level']) for r in ok})
                            / len(ok)) if ok else None,
  }


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument('--url', default='http://127.0.0.1:8011',
                 help='structured_server.py address')
  p.add_argument('--descriptions', default='data/descriptions.txt')
  p.add_argument('--env', default='MultiGrid-ReparameterizedAdversarial-v0',
                 help='A MultiGrid-*ReparameterizedAdversarial-v0 id')
  p.add_argument('--n', type=int, default=4, help='Levels per description')
  p.add_argument('--decode', choices=['sample', 'argmax'], default='sample',
                 help="Draw each cell from jev's distribution, or take its mode")
  p.add_argument('--samples', default='auto',
                 help='Server noise draws per read: "auto" or a count')
  p.add_argument('--think', type=int, default=0,
                 help='Thought tokens jev may write before each row read')
  p.add_argument('--concurrency', type=int, default=8,
                 help='Levels designed in parallel')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', default=None)
  p.add_argument('--no-render', action='store_true')
  return p.parse_args()


def main():
  args = parse_args()
  out_dir = pathlib.Path(args.out or datetime.datetime.now().strftime(
      'runs/%Y%m%d-%H%M%S'))
  (out_dir / 'levels').mkdir(parents=True, exist_ok=True)
  (out_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))

  descriptions = load_descriptions(args.descriptions)
  samples = args.samples if args.samples == 'auto' else int(args.samples)
  extensions = {'samples': samples, 'think': args.think}

  client = JevClient(args.url)
  print(f'Waiting for {args.url} ...', flush=True)
  client.wait_until_healthy()

  jobs = [(d_id, s_id) for d_id in range(len(descriptions))
          for s_id in range(args.n)]

  def design(job):
    d_id, s_id = job
    level_seed = args.seed * 100_000 + d_id * 1000 + s_id
    record = {'description_id': d_id, 'description': descriptions[d_id],
              'sample_id': s_id, 'seed': level_seed}
    env = gym.make(args.env).unwrapped
    try:
      result = designer.design_level(
          client, env, descriptions[d_id],
          rng=np.random.default_rng(level_seed), decode=args.decode,
          seed=level_seed, **extensions)
    except Exception as e:  # Keep the rest of the run going
      return {**record, 'error': repr(e)}
    record.update({
        'rows': result.rows,
        'built_level': layout.env_to_rows(env, DESIGNER_SYMBOLS),
        **layout.level_metrics(env),
        'log_likelihood': result.log_likelihood,
        'mean_entropy': result.mean_entropy,
        'requests': result.requests,
        'reads': result.reads,
        'chunks_per_row': result.chunks_per_row,
        'probabilities': result.probabilities,
    })
    if not args.no_render:
      image_path = out_dir / 'levels' / f'd{d_id:03d}_s{s_id:02d}.png'
      Image.fromarray(env.render(highlight=False)).save(image_path)
      record['image'] = str(image_path.relative_to(out_dir))
    return record

  records = []
  with open(out_dir / 'results.jsonl', 'w') as f, \
       concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
    for record in pool.map(design, jobs):
      records.append(record)
      f.write(json.dumps(record) + '\n')
      f.flush()
      status = record.get('error') or (
          f"passable={record['passable']} path={record['shortest_path_length']}"
          f" walls={record['n_walls']}")
      print(f"[{len(records)}/{len(jobs)}] d{record['description_id']} "
            f"s{record['sample_id']}: {status}", flush=True)
  client.close()

  by_description = collections.defaultdict(list)
  for r in records:
    by_description[r['description_id']].append(r)
  summary = {
      'overall': summarize(records),
      'per_description': [
          {'description': descriptions[d_id], **summarize(rs)}
          for d_id, rs in sorted(by_description.items())],
  }
  (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

  print(json.dumps(summary['overall'], indent=2))
  print(f'Wrote {len(records)} levels to {out_dir}')


if __name__ == '__main__':
  main()
