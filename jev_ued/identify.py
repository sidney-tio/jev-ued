"""jev-identify: can jev tell what a picture is? The reverse of jev-artist.

Each topic's target picture (data/topics.yaml) is shown to jev as a 13x13
0/1 grid and jev is asked which subject it shows, in one of two --mode s:

  noul    (default) one yes/no question per topic that has a target ("Does
          the picture show: heart?"), all in one /v1/systemone request. Every
          answer is p(yes) on its own, so a picture can look like several
          subjects or none; the top subject is the one with the highest
          p(yes).
  choice  one question ("What does the picture show?") whose options are
          the topics with a target. The answer is one distribution over the
          subjects, summing to 1; the top subject is its argmax. Each option
          is named by its topic id; --criteria sets its description:
            prompt  the topic's prompt (default)
            null    none, so only the id names the option
            name    the id again
            probe   the topic's probes text

Each picture is asked --k times with a different seed; with --shuffle (the
default) each repeat also lists the questions (noul) or options (choice) in
a different order, so that averaging over repeats cancels out any position
bias.

Test run (needs scripts/serve.sh running):
  python -m jev_ued.identify                    # every picture, k=5
  python -m jev_ued.identify --only heart,spiral --k 2
  python -m jev_ued.identify --mode choice      # one question, all subjects
  python -m jev_ued.identify --mode choice --criteria null

Writes to --out (default runs/identify-<timestamp>/):
  answers.jsonl   one record per picture and repeat: the mode, the state and
                  question/option order jev saw, the probability per subject
                  (p_yes for noul, p for choice), the top subject and
                  whether it is the right one
  summary.json    confusion matrix of the mean probability (picture ->
                  subject), top-1 accuracy, and the mean probability on the
                  right subject vs. the others
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

CHOICE_INSTRUCTIONS = """\
The state is a picture on a {n}x{n} grid of pixels, given as {n} rows, top to \
bottom, where 1 is a filled pixel and 0 is empty. Answer with the subject the \
picture shows."""

CHOICE_QID = 'subject'
CRITERIA = ('prompt', 'null', 'name', 'probe')


def option_description(topic, criteria):
  """Description of a topic's option in choice mode (see --criteria)."""
  if criteria == 'null':
    return None
  if criteria == 'name':
    return topic['id']
  if criteria == 'probe':
    return ' '.join(topic['probes'].split())
  return topic['prompt']


def identify_request(picture, candidates, mode='noul', criteria='prompt'):
  """State and questions in the given candidate order: one yes/no question
  per candidate (noul), or one choice over the candidates (choice) with
  option descriptions per `criteria`."""
  state = {'picture': picture}
  if mode == 'choice':
    questions = {
        CHOICE_QID: {
            'type': 'choice',
            'instructions': 'What does the picture show?',
            'criteria': {t['id']: option_description(t, criteria)
                         for t in candidates},
        }
    }
  else:
    questions = {
        t['id']: {
            'type': 'noul',
            'instructions': f"Does the picture show: {t['prompt']}?",
        }
        for t in candidates
    }
  return state, questions


def identify(client, picture, candidates, mode='noul', criteria='prompt',
             seed=0, size=SIZE, **extensions):
  """Asks jev which candidates the picture shows.

  Args:
    client: JevClient.
    picture: Rows of '0'/'1' strings.
    candidates: Topics to ask about, in question (noul) or option (choice)
      order.
    mode: 'noul' for one yes/no question per candidate, 'choice' for one
      question over all of them.
    criteria: Option descriptions in choice mode: 'prompt', 'null', 'name'
      or 'probe' (see the module docstring).
    seed: Seed for the server's noise draws.
    size: Picture side length.
    **extensions: Passed to JevClient.decide (samples, think, steps, ...).

  Returns:
    (state, {topic id: p}), p being p(yes) for noul and the choice
    probability for choice.
  """
  state, questions = identify_request(picture, candidates, mode, criteria)
  instructions = CHOICE_INSTRUCTIONS if mode == 'choice' else INSTRUCTIONS
  body = client.decide(state, questions, seed=seed,
                       instructions=instructions.format(n=size), **extensions)
  if mode == 'choice':
    return state, dict(body['answers'][CHOICE_QID]['probabilities'])
  return state, {qid: body['answers'][qid]['noul'] for qid in questions}


def prob_key(mode):
  """Record field holding the per-subject probabilities."""
  return 'p' if mode == 'choice' else 'p_yes'


def summarize(records, pictures, subjects, mode='noul', criteria=None):
  """Mean probability per (picture, subject), accuracy and diagonal vs.
  rest."""
  key = prob_key(mode)
  confusion = {
      p: {s: float(np.mean([r[key][s] for r in records
                            if r['topic_id'] == p]))
          for s in subjects}
      for p in pictures
  }
  right = [r[key][r['topic_id']] for r in records]
  wrong = [v for r in records for s, v in r[key].items()
           if s != r['topic_id']]
  return {
      'mode': mode,
      'criteria': criteria,
      'accuracy': float(np.mean([r['correct'] for r in records])),
      'accuracy_by_picture': {
          p: float(np.mean([r['correct'] for r in records
                            if r['topic_id'] == p]))
          for p in pictures
      },
      f'mean_{key}_right': float(np.mean(right)),
      f'mean_{key}_wrong': float(np.mean(wrong)),
      'confusion': confusion,
  }


def print_confusion(summary, pictures, subjects):
  width = max(map(len, pictures))
  label = 'p' if summary['mode'] == 'choice' else 'p(yes)'
  print(f'mean {label}; rows = picture shown, columns = subject asked')
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
  p.add_argument('--mode', choices=['noul', 'choice'], default='noul',
                 help='One yes/no question per subject (noul), or one '
                      'question with every subject as an option (choice)')
  p.add_argument('--criteria', choices=CRITERIA, default='prompt',
                 help='Choice mode only: describe each option by the '
                      "topic's prompt, nothing (null), its id (name), or its "
                      'probes text (probe)')
  p.add_argument('--k', type=int, default=5,
                 help='Requests per picture, each with its own seed')
  p.add_argument('--shuffle', action=argparse.BooleanOptionalAction,
                 default=True,
                 help='Shuffle the question (noul) or option (choice) order '
                      'on every request')
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
  if args.mode != 'choice':
    args.criteria = None  # Only choice options have descriptions
  elif args.criteria == 'probe':
    missing = [t['id'] for t in candidates if not t.get('probes')]
    if missing:
      raise ValueError(f'--criteria probe: no probes for {missing}')
  variant = (f'choice-{args.criteria}' if args.mode == 'choice'
             else args.mode)

  out_dir = pathlib.Path(args.out or datetime.datetime.now().strftime(
      f'runs/identify-{variant}-%Y%m%d-%H%M%S'))
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
        + (f'1 choice of {len(candidates)} subjects each'
           if args.mode == 'choice'
           else f'{len(candidates)} questions each'), flush=True)
  key = prob_key(args.mode)

  def run(job):
    topic, rep = job
    seed = args.seed * 1_000_000 + subjects.index(topic['id']) * 1000 + rep
    order = list(candidates)
    if args.shuffle:
      np.random.default_rng(seed).shuffle(order)
    state, probs = identify(client, topic['target'], order, mode=args.mode,
                            criteria=args.criteria, seed=seed,
                            samples=samples, think=args.think)
    top = max(probs, key=probs.get)
    return {
        'topic_id': topic['id'], 'repeat': rep, 'seed': seed,
        'mode': args.mode, 'criteria': args.criteria, 'state': state,
        'order': [t['id'] for t in order],
        key: {s: probs[s] for s in subjects},
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
            f"top {r['top']} ({r[key][r['top']]:.2f}), "
            f"right {r[key][r['topic_id']]:.2f}", flush=True)
  client.close()

  summary = summarize(records, pictures, subjects, args.mode, args.criteria)
  (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
  print_confusion(summary, pictures, subjects)
  label = 'p' if args.mode == 'choice' else 'p(yes)'
  print(f"Top-1 accuracy {summary['accuracy']:.2f}; mean {label} "
        f"{summary[f'mean_{key}_right']:.2f} right vs. "
        f"{summary[f'mean_{key}_wrong']:.2f} wrong")
  print(f'Answers: {out_dir / "answers.jsonl"}')


if __name__ == '__main__':
  main()
