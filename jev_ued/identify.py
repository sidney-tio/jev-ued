"""jev-identify: can jev tell what a picture is? The reverse of jev-artist.

Each topic's target picture (data/topics.yaml) is shown to jev as a 13x13
0/1 grid, with one yes/no question per topic that has a target ("Does the
picture show: heart?"), all in one /v1/systemone request. Every answer is
p(yes) on its own, so a picture can look like several subjects or none.

Each picture is asked --k times with a different seed; with --shuffle (the
default) each repeat also lists the questions in a different order, so that
averaging over repeats cancels out any position bias.

Test run (needs scripts/serve.sh running):
  python -m jev_ued.identify                    # every picture, k=5
  python -m jev_ued.identify --only heart,spiral --k 2

Writes to --out (default runs/identify-<timestamp>/):
  answers.jsonl   one record per picture and repeat: the state and question
                  order jev saw, p(yes) per subject, the top subject and
                  whether it is the right one
  summary.json    confusion matrix of mean p(yes) (picture -> subject),
                  top-1 accuracy, and mean p(yes) on the right subject vs.
                  the others
  config.json     run arguments and the topics used
"""
import argparse
import concurrent.futures
import datetime
import json
import pathlib

import numpy as np

from .artist import SIZE, load_topics
from .client import JevClient

INSTRUCTIONS = """\
The state is a picture on a {n}x{n} grid of pixels, given as {n} rows, top to \
bottom, where 1 is a filled pixel and 0 is empty. Each question names a \
subject: answer yes if the picture shows that subject, and no otherwise."""


def identify_request(picture, candidates):
  """State and one yes/no question per candidate topic, in the given order."""
  state = {'picture': picture}
  questions = {
      t['id']: {
          'type': 'noul',
          'instructions': f"Does the picture show: {t['prompt']}?",
      }
      for t in candidates
  }
  return state, questions


def identify(client, picture, candidates, seed=0, size=SIZE, **extensions):
  """Asks jev which candidates the picture shows.

  Args:
    client: JevClient.
    picture: Rows of '0'/'1' strings.
    candidates: Topics to ask about, in question order.
    seed: Seed for the server's noise draws.
    size: Picture side length.
    **extensions: Passed to JevClient.decide (samples, think, steps, ...).

  Returns:
    (state, {topic id: p(yes)}).
  """
  state, questions = identify_request(picture, candidates)
  body = client.decide(state, questions, seed=seed,
                       instructions=INSTRUCTIONS.format(n=size), **extensions)
  return state, {qid: body['answers'][qid]['noul'] for qid in questions}


def summarize(records, pictures, subjects):
  """Mean p(yes) per (picture, subject), accuracy and diagonal vs. rest."""
  confusion = {
      p: {s: float(np.mean([r['p_yes'][s] for r in records
                            if r['topic_id'] == p]))
          for s in subjects}
      for p in pictures
  }
  right = [r['p_yes'][r['topic_id']] for r in records]
  wrong = [v for r in records for s, v in r['p_yes'].items()
           if s != r['topic_id']]
  return {
      'accuracy': float(np.mean([r['correct'] for r in records])),
      'accuracy_by_picture': {
          p: float(np.mean([r['correct'] for r in records
                            if r['topic_id'] == p]))
          for p in pictures
      },
      'mean_p_yes_right': float(np.mean(right)),
      'mean_p_yes_wrong': float(np.mean(wrong)),
      'confusion': confusion,
  }


def print_confusion(summary, pictures, subjects):
  width = max(map(len, pictures))
  print('mean p(yes); rows = picture shown, columns = subject asked')
  print(' ' * width + ' ' + ' '.join(f'{s[:6]:>6}' for s in subjects))
  for p in pictures:
    print(f'{p:>{width}} ' + ' '.join(
        f"{summary['confusion'][p][s]:6.2f}" for s in subjects))


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument('--dataset', default='data/topics.yaml',
                 help='YAML file of topics; only topics with a target are '
                      'used')
  p.add_argument('--only', default=None,
                 help='Comma-separated topic ids whose pictures to show '
                      '(default: all). Every topic with a target is still '
                      'asked about.')
  p.add_argument('--k', type=int, default=5,
                 help='Requests per picture, each with its own seed')
  p.add_argument('--shuffle', action=argparse.BooleanOptionalAction,
                 default=True,
                 help='Shuffle the question order on every request')
  p.add_argument('--url', default='http://127.0.0.1:8011',
                 help='structured_server.py address')
  p.add_argument('--samples', default='auto',
                 help='Server noise draws per read: "auto" or a count')
  p.add_argument('--think', type=int, default=0,
                 help='Thought tokens jev may write before answering')
  p.add_argument('--concurrency', type=int, default=8,
                 help='Requests in flight at once')
  p.add_argument('--instructions-in-state', action='store_true',
                 help='Send the instructions inside the state, for servers '
                      'that ignore a top-level instructions field (kev.serve)')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', default=None)
  return p.parse_args()


def main():
  args = parse_args()
  candidates = [t for t in load_topics(args.dataset) if 'target' in t]
  subjects = [t['id'] for t in candidates]
  if args.only:
    only = args.only.split(',')
    unknown = set(only) - set(subjects)
    if unknown:
      raise ValueError(f'no topic with a target for ids: {sorted(unknown)}')
    shown = [t for t in candidates if t['id'] in only]
  else:
    shown = candidates
  pictures = [t['id'] for t in shown]

  out_dir = pathlib.Path(args.out or datetime.datetime.now().strftime(
      'runs/identify-%Y%m%d-%H%M%S'))
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / 'config.json').write_text(json.dumps(
      {**vars(args), 'topics': candidates}, indent=2))
  samples = args.samples if args.samples == 'auto' else int(args.samples)

  client = JevClient(args.url,
                     instructions_in_state=args.instructions_in_state)
  print(f'Waiting for {args.url} ...', flush=True)
  client.wait_until_healthy()

  jobs = [(t, rep) for t in shown for rep in range(args.k)]
  print(f'{len(shown)} pictures x k={args.k} = {len(jobs)} requests, '
        f'{len(candidates)} questions each', flush=True)

  def run(job):
    topic, rep = job
    seed = args.seed * 1_000_000 + subjects.index(topic['id']) * 1000 + rep
    order = list(candidates)
    if args.shuffle:
      np.random.default_rng(seed).shuffle(order)
    state, p_yes = identify(client, topic['target'], order, seed=seed,
                            samples=samples, think=args.think)
    top = max(p_yes, key=p_yes.get)
    return {
        'topic_id': topic['id'], 'repeat': rep, 'seed': seed,
        'state': state, 'order': [t['id'] for t in order],
        'p_yes': {s: p_yes[s] for s in subjects},
        'top': top, 'correct': top == topic['id'],
    }

  records = []
  with open(out_dir / 'answers.jsonl', 'w') as f, \
       concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
    futures = [pool.submit(run, job) for job in jobs]
    for future in concurrent.futures.as_completed(futures):
      r = future.result()
      records.append(r)
      f.write(json.dumps(r) + '\n')
      f.flush()
      print(f"[{len(records)}/{len(jobs)}] {r['topic_id']} r{r['repeat']:02d}: "
            f"top {r['top']} ({r['p_yes'][r['top']]:.2f}), "
            f"right {r['p_yes'][r['topic_id']]:.2f}", flush=True)
  client.close()

  summary = summarize(records, pictures, subjects)
  (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
  print_confusion(summary, pictures, subjects)
  print(f"Top-1 accuracy {summary['accuracy']:.2f}; mean p(yes) "
        f"{summary['mean_p_yes_right']:.2f} right vs. "
        f"{summary['mean_p_yes_wrong']:.2f} wrong")
  print(f'Answers: {out_dir / "answers.jsonl"}')


if __name__ == '__main__':
  main()
