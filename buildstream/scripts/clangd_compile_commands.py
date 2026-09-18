#!/usr/bin/env python3

"""Rewrite the compile commands of a build tree into ones clangd can use.

Usage: clangd_compile_commands.py <build>/compile_commands.json <output>

A unity build compiles the sources of a target through generated
unity_<n>_cxx.cxx files, and compile_commands.json only names those. clangd then
guesses the flags of a source from a neighbour, which may belong to another
target or be compiled with a standard of its own. Each unity entry is expanded
back into one entry per source it includes.

The precompiled header flags go too: the .pch is only valid for the compiler
that built it, and goes stale as soon as a header is edited. The header it was
made from is still force-included, since the sources may rely on what it
declares.
"""

import json
import os
import re
import shlex
import sys
import tempfile

UNITY_FILE = re.compile(r"/CMakeFiles/[^/]+\.dir/Unity/unity_[^/]+$")
UNITY_INCLUDE = re.compile(r'^#include "(.+)"$', re.MULTILINE)


def without_pch(args):
    """Drop '-Xclang -include-pch -Xclang <pch>' and turn
    '-Xclang -include -Xclang <header>' into '-include <header>'."""
    result = []
    i = 0
    while i < len(args):
        if (
            args[i] == "-Xclang"
            and i + 3 < len(args)
            and args[i + 1] in ("-include-pch", "-include")
            and args[i + 2] == "-Xclang"
        ):
            if args[i + 1] == "-include":
                result += ["-include", args[i + 3]]
            i += 4
        else:
            result.append(args[i])
            i += 1
    return result


def rewrite(entry):
    """Yield the entries clangd should see in place of this one."""
    path = os.path.join(entry["directory"], entry["file"])
    unity = UNITY_FILE.search(path) is not None
    command = entry.get("command")
    if command is not None and not unity and "-include-pch" not in command:
        yield entry
        return

    args = entry["arguments"] if command is None else shlex.split(command)
    args = without_pch(args)
    if not unity:
        yield {**{k: v for k, v in entry.items() if k != "command"}, "arguments": args}
        return

    try:
        with open(path, encoding="utf-8") as f:
            sources = UNITY_INCLUDE.findall(f.read())
    except OSError:
        sources = []
    if not sources:
        yield {"directory": entry["directory"], "file": entry["file"], "arguments": args}
        return
    for source in sources:
        yield {
            "directory": entry["directory"],
            "file": source,
            "arguments": [source if arg == path else arg for arg in args],
        }


def main():
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <build>/compile_commands.json <output>")
    source, output = sys.argv[1:]

    with open(source, encoding="utf-8") as f:
        entries = json.load(f)
    rewritten = [new for entry in entries for new in rewrite(entry)]

    # clangd reloads the file whenever it changes, so never let it see a partial
    # one; and each clangd session refreshes it, possibly at the same time.
    directory = os.path.dirname(os.path.abspath(output))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".json.tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(rewritten, f, indent=1)
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
