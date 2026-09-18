#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic file writes for long-running unattended experiments (E3+).

Guards against partial files from a crash or power loss mid-write: the
caller's content is written to a temp file in the same directory as the
destination, fsynced, atomically renamed into place with os.replace(),
and the parent directory is then fsynced too. The directory fsync
matters on its own -- without it, the file's data can survive a power
loss while the directory entry that makes the rename visible does not,
leaving the final path missing even though the data was written.
"""
import os
import tempfile
from pathlib import Path


def atomic_write(final_path, write_fn, mode='w', newline=None, encoding=None):
    """Write content to final_path atomically.

    write_fn(fileobj) must write the complete content to the given open
    file object. On success, final_path is replaced by a single atomic
    rename and the parent directory is fsynced so the rename itself
    survives a power loss. On any failure the temp file is removed and
    final_path is left untouched.

    Parameters
    ----------
    final_path : str or Path
        Destination file path. Its parent directory is created if
        missing and must end up on the same filesystem as the temp
        file for the rename to be atomic (guaranteed here since the
        temp file is created in that same parent directory).
    write_fn : callable
        Called once as write_fn(fileobj) with the open temp file.
    mode, newline, encoding : passed through to os.fdopen().

    Returns
    -------
    str
        The final path written.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(final_path.parent),
        prefix=f'.{final_path.name}.',
        suffix='.tmp',
    )
    try:
        with os.fdopen(fd, mode, newline=newline, encoding=encoding) as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(final_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    dir_fd = os.open(str(final_path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    return str(final_path)


def atomic_rename(src, dst):
    """Atomically move an already-written file from src to dst.

    Used to relocate a file that some other function already wrote (e.g.
    perform_evaluate.py's save_* functions, which always derive their
    filename from a caller-supplied basename and cannot be told to write
    directly into a nested directory) into its final home without a
    window where a partially-relocated file could be observed.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src), str(dst))

    dir_fd = os.open(str(dst.parent), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    return str(dst)
