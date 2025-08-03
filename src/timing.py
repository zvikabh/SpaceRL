import datetime


class Timer:
  """Context manager which shows the runtime of whatever is inside its context."""

  def __init__(self, title: str = '', disable: bool = False):
    self.title = title
    self.disable = disable

  def __enter__(self):
    self.start_time = datetime.datetime.now()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not self.disable:
      elapsed = datetime.datetime.now() - self.start_time
      print(f'{self.title} completed in {elapsed}')
