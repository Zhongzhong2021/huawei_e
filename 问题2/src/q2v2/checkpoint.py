"""Transactional checkpoint writes; an interrupted save never truncates best.pt.

This protects the previous checkpoint, not arbitrary mid-epoch resumption.
Best-model records do not contain optimizer or RNG state. A failed run must be
replayed from its original initialization in a NEW directory, not resumed as if
those states were available. Same-directory atomic replacement is required.
"""
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import torch


def cpu_snapshot(value):
    """Finish synchronous GPU transfers before creating a checkpoint file."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        out = OrderedDict() if isinstance(value, OrderedDict) else {}
        out.update((k, cpu_snapshot(v)) for k, v in value.items())
        if hasattr(value, '_metadata'):
            out._metadata = copy.deepcopy(value._metadata)
        return out
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def equal_record(expected, actual, location='record'):
    """Verify metadata, tensor shapes/dtypes and every value after deserialization."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.device.type != 'cpu':
            raise ValueError(f'{location}: expected CPU tensor')
        if expected.dtype != actual.dtype or expected.shape != actual.shape or not torch.equal(expected, actual):
            raise ValueError(f'{location}: tensor roundtrip mismatch')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or list(expected) != list(actual):
            raise ValueError(f'{location}: mapping keys mismatch')
        if getattr(expected, '_metadata', None) != getattr(actual, '_metadata', None):
            raise ValueError(f'{location}: state-dict metadata mismatch')
        for k in expected:
            equal_record(expected[k], actual[k], f'{location}.{k}')
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(expected) != len(actual):
            raise ValueError(f'{location}: sequence mismatch')
        for i, (a, b) in enumerate(zip(expected, actual)):
            equal_record(a, b, f'{location}[{i}]')
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f'{location}: value mismatch')


def atomic_save(record, destination, *, event_path=None):
    """Save CPU state, fsync, reload/compare, then atomically replace destination.

    An exception leaves the previous destination intact before replacement and
    retains the uniquely named temporary file for diagnosis. Hard termination
    likewise may leave a temporary file, never an intentionally partial best.pt.
    A failure AFTER replacement is recorded as such; it is not rolled back.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    temporary = None
    committed = False

    def event(phase, **extra):
        if event_path is None:
            return
        data = {'utc': datetime.now(timezone.utc).isoformat(), 'phase': phase,
                'epoch': record.get('epoch'), 'destination': destination.name,
                'temporary': temporary.name if temporary else None,
                'seconds': time.monotonic()-started, **extra}
        with Path(event_path).open('a', encoding='utf-8') as log:
            log.write(json.dumps(data, allow_nan=False)+'\n')
            log.flush()
            os.fsync(log.fileno())

    try:
        event('cpu_snapshot_started')
        snapshot = cpu_snapshot(record)
        event('cpu_snapshot_completed')
        fd, name = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
        temporary = Path(name)
        with os.fdopen(fd, 'wb') as stream:
            event('serialize_started')
            torch.save(snapshot, stream)
            stream.flush()
            os.fsync(stream.fileno())
        event('serialize_completed', bytes=temporary.stat().st_size)
        loaded = torch.load(temporary, map_location='cpu', weights_only=True)
        equal_record(snapshot, loaded)
        del loaded
        event('verified')
        os.replace(temporary, destination)
        committed = True
        if os.name == 'posix':
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        event('committed', bytes=destination.stat().st_size)
    except BaseException as exc:
        try:
            event('failed', committed=committed, error=repr(exc))
        except OSError:
            pass  # Do not replace the underlying storage error with a log error.
        raise
