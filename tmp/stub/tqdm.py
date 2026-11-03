"""Minimal tqdm stub: the metric functions must stay byte-identical, so the
progress bar only is replaced locally."""


def tqdm(iterable, **kwargs):
    return iterable
