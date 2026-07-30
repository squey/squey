#!/usr/bin/env python3

"""Convert a cppcheck XML report (version 2) to the GitLab Code Quality format.

This replaces the CodeClimate cppcheck plugin, which required Docker-in-Docker
and is not usable on Podman based runners.
"""

import hashlib
import json
import sys
import xml.etree.ElementTree as ET

# cppcheck severities mapped to the ones understood by GitLab
SEVERITIES = {
    "error": "major",
    "warning": "minor",
    "performance": "minor",
    "portability": "minor",
    "style": "minor",
    "information": "info",
    "debug": "info",
}

CATEGORIES = {
    "error": "Bug Risk",
    "warning": "Bug Risk",
    "performance": "Performance",
    "portability": "Compatibility",
    "style": "Style",
    "information": "Clarity",
    "debug": "Clarity",
}


def parse_report(xml_path):
    """Read a cppcheck report, refusing any DTD.

    ElementTree expands internal entities, so a report carrying a DTD could
    exhaust the memory of the job. cppcheck never emits one.
    """
    with open(xml_path, "rb") as report:
        content = report.read()

    if b"<!DOCTYPE" in content:
        sys.exit(f"{xml_path}: unexpected DTD in a cppcheck report")

    # The DTD rejected above is what makes entity expansion possible
    # nosemgrep
    return ET.fromstring(content)


def convert(xml_path):
    issues = []
    occurrences = {}

    for error in parse_report(xml_path).iter("error"):
        # cppcheck lists the primary location first, the following ones only
        # describe how the code flows up to it
        location = error.find("location")
        if location is None:
            continue

        severity = error.get("severity")
        check_name = error.get("id")
        description = error.get("msg")
        path = location.get("file")

        # Keep the fingerprint stable when code moves around, while staying
        # unique for identical messages reported several times in a file
        key = f"{path}:{check_name}:{description}"
        occurrences[key] = occurrences.get(key, 0) + 1
        fingerprint = hashlib.sha256(f"{key}:{occurrences[key]}".encode()).hexdigest()

        issues.append(
            {
                "type": "issue",
                "check_name": f"cppcheck.{check_name}",
                "description": description,
                "categories": [CATEGORIES.get(severity, "Bug Risk")],
                "severity": SEVERITIES.get(severity, "info"),
                "fingerprint": fingerprint,
                "location": {
                    "path": path,
                    "lines": {"begin": max(int(location.get("line") or 1), 1)},
                },
            }
        )

    return issues


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <cppcheck.xml> <gl-code-quality-report.json>")

    issues = convert(sys.argv[1])
    with open(sys.argv[2], "w") as report:
        json.dump(issues, report, indent=2)

    print(f"{len(issues)} issue(s) reported in {sys.argv[2]}")
