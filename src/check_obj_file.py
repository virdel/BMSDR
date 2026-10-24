"""Helpers for preparing OBJ files before they are read by CGAL / the trainer.

The old implementation loaded the whole file into memory, guessed the text
encoding and then always rewrote the file as UTF-8.  Rewriting an ANSI/GBK
model as UTF-8 could silently corrupt it.  The helpers below detect the
encoding once, stream the file and keep the original encoding.
"""

from __future__ import print_function

import argparse
import os
import sys

TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "latin-1")


def detect_encoding(filename):
    """Return the first encoding from :data:`TEXT_ENCODINGS` that can decode it."""
    with open(filename, "rb") as handle:
        data = handle.read()
    for encoding in TEXT_ENCODINGS:
        try:
            data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def get_vertex_triangle_num(filename, encoding=None):
    """Count ``v`` and ``f`` records without loading the whole file."""
    encoding = encoding or detect_encoding(filename)
    vertex_numbers = 0
    face_numbers = 0
    with open(filename, "r", encoding=encoding) as handle:
        for line in handle:
            if line.startswith("v "):
                vertex_numbers += 1
            elif line.startswith("f "):
                face_numbers += 1
    return vertex_numbers, face_numbers


def check_obj_file(filename, encoding=None):
    """Remove ``l`` (polyline) records that CGAL cannot read.

    The file is only rewritten when at least one line was removed, and it
    keeps its original encoding and line endings.  Returns the number of
    removed records.
    """
    filename = os.path.abspath(filename)
    encoding = encoding or detect_encoding(filename)
    temporary = filename + ".tmp"
    removed = 0

    try:
        with open(filename, "r", encoding=encoding, newline="") as source:
            with open(temporary, "w", encoding=encoding, newline="") as target:
                for line in source:
                    if line.startswith("l "):
                        removed += 1
                        continue
                    target.write(line)
        if removed:
            os.replace(temporary, filename)
            print("Removed %d line record(s) from %s" % (removed, filename))
        else:
            os.remove(temporary)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise

    return removed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a mesh before simplification")
    parser.add_argument("mesh", help="OBJ file")
    parser.add_argument(
        "--count",
        action="store_true",
        help="only print vertex/face counts, do not modify the file",
    )
    args = parser.parse_args(argv)

    if args.count:
        vertices, faces = get_vertex_triangle_num(args.mesh)
        print("vertices=%d faces=%d" % (vertices, faces))
    else:
        print("removed=%d" % check_obj_file(args.mesh))
    return 0


if __name__ == "__main__":
    sys.exit(main())
