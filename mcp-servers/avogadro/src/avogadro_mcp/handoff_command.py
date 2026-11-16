"""Avogadro 2 command "Extensions > CLIO > Send to CLIO", written into the session.

Avogadro runs it with the molecule as CJSON on stdin. It writes that CJSON (with the
person's selection, ``atoms.selected``) into the handoff folder the server watches, and
returns the molecule unchanged: without ``cjson`` in its reply Avogadro replaces the
molecule with an empty one.
"""

import argparse
import json
import os
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser()
    for flag in ("--print-options", "--display-name", "--menu-path", "--run-command"):
        p.add_argument(flag, action="store_true")
    p.add_argument("--lang", nargs="?", default="en")
    a = p.parse_args()
    if a.print_options:
        print(json.dumps({"inputMoleculeFormat": "cjson"}))
    elif a.display_name:
        print("Send to CLIO")
    elif a.menu_path:
        print("&Extensions|CLIO")
    else:
        data = json.loads(sys.stdin.read())
        folder = os.environ["CLIO_AVOGADRO_HANDOFF"]
        os.makedirs(folder, exist_ok=True)
        name = f"sent-{time.time_ns()}.cjson"
        tmp = os.path.join(folder, name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(data.get("cjson", data), fh)
        os.replace(tmp, os.path.join(folder, name))
        print(json.dumps({"message": "Sent to CLIO.", "cjson": data.get("cjson")}))


if __name__ == "__main__":
    main()
