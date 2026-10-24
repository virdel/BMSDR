"""Mesh simplification helpers used by :mod:`train`.

The repository still ships the QEM command line tool in ``bin/qem``.  This
module wraps it so the training pipeline does not depend on the current
working directory and does not build shell commands by string
concatenation (paths with spaces used to break the old ``os.system``
call).
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QEM_EXE = os.path.join(REPO_ROOT, "bin", "qem", "qem.exe")
QEM_OUTPUT_NAME = "mesh_qem_3000.obj"


def qem_simplify(input_obj, output_dir, exe=None, timeout=None):
    """Simplify ``input_obj`` with the bundled QEM tool.

    The executable writes ``mesh_qem_3000.obj`` into ``output_dir``.  The
    function returns ``(return_code, output_path)``; the output path is
    returned even when the tool failed so callers can log it.
    """
    input_obj = os.path.abspath(input_obj)
    output_dir = os.path.abspath(output_dir)
    executable = os.path.abspath(exe) if exe else DEFAULT_QEM_EXE

    if not os.path.isfile(input_obj):
        raise IOError("input mesh not found: %s" % input_obj)
    if not os.path.isfile(executable):
        raise IOError(
            "QEM executable not found: %s (pass --qem-exe or build bin/qem)"
            % executable
        )

    os.makedirs(output_dir, exist_ok=True)
    command = [executable, input_obj, output_dir]
    print("Running: %s" % " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        print(output)
        raise RuntimeError("QEM simplification timed out after %s s" % timeout)

    if output:
        print(output.rstrip())

    return process.returncode, os.path.join(output_dir, QEM_OUTPUT_NAME)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simplify an OBJ mesh with QEM")
    parser.add_argument("input", help="input OBJ file")
    parser.add_argument("output_dir", help="directory for mesh_qem_3000.obj")
    parser.add_argument("--qem-exe", default=None, help="override the QEM executable")
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args(argv)

    status, output_path = qem_simplify(
        args.input, args.output_dir, exe=args.qem_exe, timeout=args.timeout
    )
    print("exit=%d output=%s" % (status, output_path))
    return status


if __name__ == "__main__":
    sys.exit(main())
