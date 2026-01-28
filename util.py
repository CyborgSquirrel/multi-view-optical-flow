import contextlib as ctl
import dataclasses as dc
import functools as ft
import glob
import hashlib
import itertools as itt
import logging
import os
import os.path as osp
import subprocess
import threading
import time
import urllib
from typing import Any, Callable, Generic, Optional, TypeVar

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
_logger = logger

REPO_ROOT = osp.dirname(__file__)

def pipe(a, *fns):
  for fn in fns:
    a = fn(a)
  return a

def proc(
  executable: str,
  *args,
  check=True, env=None,
  stdout=False, stderr=False,
  cwd=None,
) -> Optional[str]:
  if sum([stdout is True, stderr is True]) > 1:
    raise ValueError(f"Only one of {stdout=}, {stderr=}, may be set to {repr(True)}")
  
  kwargs = {}
  if stdout is not False:
    kwargs["stdout"] = subprocess.PIPE
  if stderr is not False:
    kwargs["stderr"] = subprocess.PIPE

  res = subprocess.run([executable, *args], check=check, env=env, cwd=cwd, **kwargs)

  if stdout is True:
    return res.stdout.decode()
  if stderr is True:
    return res.stderr.decode()

  return None

DL_CACHE_DIR = osp.join(REPO_ROOT, 'downloads.gen')
def dl_cached(url: str) -> str:
  """Downloads file at `url` and caches it locally."""

  url_parsed = urllib.parse.urlparse(url)
  h = hashlib.sha256(url.encode()).digest()[:16].hex()
  name = h + '-' + osp.basename(url_parsed.path)
  path = osp.join(DL_CACHE_DIR, name)

  if osp.exists(path):
    logger.info('Using cached %r', url)
  else:
    os.makedirs(DL_CACHE_DIR, exist_ok=True)

    temp_path = path + '.temp'
    logger.info('Downloading %r...', url)
    with ctl.ExitStack() as stack:
      r = stack.enter_context(requests.get(url, stream=True))
      f = stack.enter_context(open(temp_path, 'wb'))

      size = r.headers.get('Content-Length')
      if size is not None:
        size = int(size)

      pbar = None
      if size is not None:
        pbar = stack.enter_context(tqdm(desc='Downloading...', total=size))

      for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
        pbar.update(len(chunk))
    logger.info('Finished downloading %r', url)
    os.rename(temp_path, path)
  return path

def resolve_local_path(path_or_url: str) -> str:
  parsed = urllib.parse.urlparse(path_or_url)
  if parsed.netloc == "":
    return parsed.path
  return dl_cached(path_or_url)

def iglob(pattern: str, strict: bool = True):
  '''Version of `glob.iglob` which makes sure that at least one file matches the pattern.'''

  paths = glob.iglob(pattern, recursive=True)
  if strict:
    try:
      yield next(paths)
    except StopIteration as ex:
      raise RuntimeError(f'File pattern {repr(pattern)} didn\'t match any files.') from ex
  yield from paths

T = TypeVar('T')

class Oneshot(Generic[T]):
  inner: T

  def __init__(self):
    self.lock = threading.Lock()
    self.event = threading.Event()

  def put(self, val: T):
    with self.lock:
      if hasattr(self, 'inner'):
        raise RuntimeError('Value has already been put')
      self.inner = val
      self.event.set()

  def get(self) -> T:
    self.event.wait()
    with self.lock:
      return self.inner

class ThreadPool:
  def __init__(self, max_workers=8):
    self.max_workers = max_workers
    self.sem = threading.Semaphore(max_workers)
    self.lock = threading.Lock()
    self.open = False

  def __wrap(self, fn):
    def wrapped(*args, **kwargs):
      try:
        return fn(*args, **kwargs)
      finally:
        self.sem.release()
    return wrapped

  def submit(self, fn, *args, **kwargs):
    with self.lock:
      if not self.open:
        raise RuntimeError(f"Can't submit task when {self.open=}")

    self.sem.acquire()
    thread = threading.Thread(
      target=self.__wrap(fn),
      args=args, kwargs=kwargs,
    )
    thread.start()

  def __enter__(self):
    with self.lock:
      self.open = True
    return self

  def __exit__(self, ex_type, ex_val, ex_tb):
    with self.lock:
      self.open = False

    # Wait for all workers to finish
    for _ in range(self.max_workers):
      self.sem.acquire()
    self.sem.release(self.max_workers)

class PrefixLogger(logging.LoggerAdapter):
  def __init__(self, logger, prefix):
    super().__init__(logger, dict(prefix=prefix))
    self.prefix = prefix

  def process(self, msg, kwargs):
    return f'{self.prefix}{msg}', kwargs

def make_setup_subparser(subparsers):
  def setup_subparser(setup_subparser_fn: Optional[Callable] = None):
    def decorated(build_fn: Callable):
      fn_name: str = build_fn.__name__

      # Check prefix exists, and remove it
      prefix = 'tool_'
      if not fn_name.startswith(prefix):
        raise ValueError(f'Function name {repr(fn_name)} must start with {repr(prefix)}')
      fn_name = fn_name.removeprefix(prefix)
    
      # Set up subparser
      subparser_name = fn_name.replace('_', '-')
      subparser = subparsers.add_parser(subparser_name, description=build_fn.__doc__)
      subparser.set_defaults(fn=build_fn)
      if setup_subparser_fn is not None:
        setup_subparser_fn(subparser)

      return build_fn
    if setup_subparser_fn is not None:
      decorated = ft.wraps(setup_subparser_fn)(decorated)
    return decorated
  return setup_subparser

############################################################
#                       Dependencies                       #
############################################################

class CondInp:
  '''Conditional input for deps.'''
  def __init__(
    self,
    cond: Callable,
    inps: list[str | Callable],
  ):
    self.cond = cond
    self.inps = inps

class CancelBuild(Exception):
  def __init__(self):
    self.add_note(f'Possibly you forgot to use {deps.__name__}.outer')

class AlreadyRan(CancelBuild):
  pass

class OutputsFresh(CancelBuild):
  pass

# NOTE: When used as a decorator, it eats the return value of the function.
class deps: # pylint: disable=invalid-name
  outer = ctl.suppress(CancelBuild)

  name: Optional[str]
  
  def __init__(
     self,
     outs: str | list[str],
     inps: list[str | Callable | CondInp],
     *,
     name: Optional[str] = None,
  ):
    # Ensure out is a list
    if isinstance(outs, str):
      outs = [outs]

    self._t0_all = None
    self._t0_me = None
    self._t1 = None
    self._outdated = None
    self._ran = False
    self.name = name
    self.inps = inps
    self.outs = outs

  @classmethod
  def _expand_conds(cls, inps):
    for inp in inps:
      if isinstance(inp, CondInp):
        if not inp.cond():
          continue
        yield from cls._expand_conds(inp.inps)
      else:
        yield inp

  @ft.cached_property
  def _logger(self):
    logger = _logger
    if self.name is not None:
      logger = PrefixLogger(logger, f'[{self.name}] ')
    return logger

  def _check_outdated(self):
    file_inps = []
    fn_inps = []

    for inp in self._expand_conds(self.inps):
      if callable(inp):
        fn_inps.append(inp)
      elif isinstance(inp, str):
        file_inps.append(inp)
    
    # Run all functions
    for fn_inp in fn_inps:
      fn_inp()

    # Make one big iterator with files
    file_inps = list(itt.chain.from_iterable(
      iglob(inp)
      for inp in file_inps
    ))

    # All output files
    out_mtime_min = None
    for out in self.outs:
      if not osp.exists(out):
        self._logger.info('Output file %r doesn\'t exist, rebuilding...', out)
        return True

      stat = os.stat(out)
      if out_mtime_min is None or stat.st_mtime < out_mtime_min:
        out_mtime_min = stat.st_mtime

    if out_mtime_min is None:
      self._logger.info('No output files, rebuilding...')
      return True

    # All input files
    for inp in file_inps:
      if not osp.exists(inp):
        raise RuntimeError(f'Input file {repr(inp)} doesn\'t exist.')
      inp_stat = os.stat(inp)

      if inp_stat.st_mtime > out_mtime_min:
        self._logger.info('Input file %r changed, rebuilding...', inp)
        return True

    self._logger.info('Inputs staid the same, skipping.')
    return False

  def __call__(self, fn):
    if self.name is None:
      self.name = fn.__name__

    @ft.wraps(fn)
    def decorated(*args, **kwargs):
      with self.outer, self:
        ret = fn(*args, **kwargs)
        if ret is not None:
          raise RuntimeError(
            f'Function decorated with \'@{self.__class__.__name__}\' must not return anything.'
          )
    return decorated

  def __enter__(self):
    if self._ran:
      raise AlreadyRan()

    self._ran = True
    self._t0_all = time.time()
    if not self._check_outdated():
      raise OutputsFresh()

    self._t0_me = time.time()
    return self

  def __exit__(self, ex_type, ex_val, ex_tb):
    self._t1 = time.time()
    self._logger.info(
      'Rule took: %.2fs/%.2fs.',
      self._t1-self._t0_me,
      self._t1-self._t0_all,
    )

    for out in self.outs:
      if not osp.exists(out):
        raise RuntimeError(f'Output file {repr(out)} doesn\'t exist after building.')
