"""Python client for jev's decision API (/v1/systemone).

structured_server.py serves the API in front of vLLM. A request carries a
`state` (any JSON) and `questions`, a map of id -> {type, instructions,
criteria}. Each answer is a probability distribution over the question's
options, read from one denoising step of DiffusionGemma.
"""
import os
import time

import httpx


class JevError(RuntimeError):
  """The server rejected or failed a request."""


class JevClient:
  """Thread-safe client; share one instance across worker threads."""

  def __init__(self, base_url='http://127.0.0.1:8011', model='jev-latest',
               api_key=None, timeout=600.0, retries=3):
    api_key = api_key or os.environ.get('JEV_API_KEY')
    headers = {'authorization': f'Bearer {api_key}'} if api_key else {}
    self.http = httpx.Client(base_url=base_url, headers=headers,
                             timeout=timeout)
    self.model = model
    self.retries = retries

  def wait_until_healthy(self, timeout=600.0, interval=2.0):
    """Blocks until the server's /health answers, e.g. while vLLM loads."""
    deadline = time.monotonic() + timeout
    while True:
      try:
        if self.http.get('/health').status_code == 200:
          return
      except httpx.TransportError:
        pass
      if time.monotonic() > deadline:
        raise TimeoutError(f'{self.http.base_url} not healthy after {timeout}s')
      time.sleep(interval)

  def decide(self, state, questions, seed=None, **extensions):
    """Asks jev the questions about state.

    Args:
      state: JSON-serializable state, shown to the model as the user message.
      questions: Map of question id -> {'type': 'noul'|'choice'|'score',
        'instructions': str, 'criteria': ...}.
      seed: Seed for the server's noise draws.
      **extensions: Server schema options, e.g. instructions, samples, think,
        steps, sequential.

    Returns:
      The response body: {'answers': {id: answer}, 'usage', 'diagnostics'}.
      A choice answer is {'choice', 'probabilities': {option: p},
      'confidence'}.
    """
    body = {'model': self.model, 'state': state, 'questions': questions,
            **extensions}
    if seed is not None:
      body['seed'] = int(seed)

    for attempt in range(self.retries + 1):
      try:
        response = self.http.post('/v1/systemone', json=body)
      except httpx.TransportError as e:
        if attempt == self.retries:
          raise JevError(f'request failed: {e!r}') from e
      else:
        if response.status_code == 200:
          return response.json()
        # 4xx is a bad schema/request: retrying won't help
        if response.status_code < 500 or attempt == self.retries:
          raise JevError(f'{response.status_code}: {response.text[:500]}')
      time.sleep(2 ** attempt)

  def close(self):
    self.http.close()
