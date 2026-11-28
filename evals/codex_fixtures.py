"""Disposable inputs and independent artifact checks for live Codex evaluations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]


def git(project, *args):
    return subprocess.run(
        ["git", *args], cwd=project, capture_output=True, text=True, check=True
    ).stdout


def prepare(project: Path, case: dict) -> dict[str, str]:
    project.mkdir(parents=True)
    files = {
        "README.md": "# Scientific research fixture\nLocal inputs for bounded analysis tasks. No dependencies or remote.\n",
        "runs.csv": "machine,size,runtime\nalpha,1,10\nalpha,2,20\nalpha,2,22\nbeta,1,8\nbeta,2,16\nbeta,4,\n",
        "gaps.csv": "time,value\n0,1\n1,\n2,3\n3,\n4,5\n",
        "waveform.csv": "time,amplitude\n0,0\n1,1\n2,0\n3,-1\n4,0\n",
        "app.log": "2026-09-21 12:00:00 INFO starting\n2026-09-21 12:01:00 ERROR storage timeout\n2026-09-21 12:01:01 ERROR retry exhausted\n2026-09-21 12:02:00 INFO ERROR counter reset\n",
    }
    for name, content in files.items():
        p = project / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    polygon = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
        },
    }
    points = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": label},
                "geometry": {"type": "Point", "coordinates": xy},
            }
            for label, xy in (("inside", [1, 1]), ("outside", [3, 3]))
        ],
    }
    (project / "region.geojson").write_text(json.dumps(polygon))
    (project / "points.geojson").write_text(json.dumps(points))
    (project / "field.vti").write_text(
        '<?xml version="1.0"?><VTKFile type="ImageData" version="0.1" byte_order="LittleEndian"><ImageData WholeExtent="0 1 0 1 0 1" Origin="0 0 0" Spacing="1 1 1"><Piece Extent="0 1 0 1 0 1"><PointData Scalars="temperature"><DataArray type="Float64" Name="temperature" format="ascii">0 1 2 3 4 5 6 7</DataArray></PointData><CellData/></Piece></ImageData></VTKFile>'
    )
    if "hdf5" in case["servers"]:
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "--project",
                str(ROOT / "mcp-servers/hdf5"),
                "python",
                "-c",
                "import h5py,numpy as np,sys\nwith h5py.File(sys.argv[1],'w') as f:\n"
                + (
                    " d=f.create_dataset('temperature',shape=(70000000,),dtype='f8',chunks=(10000,),fillvalue=1.0);d[1]=1000;d.attrs['units']='K'\n"
                    if case["skill"] == "large-data-read"
                    else " d=f.create_dataset('temperature',data=np.arange(1000,dtype=float));d.attrs['units']='K'\n"
                )
                + " d=f.create_dataset('readings',data=np.column_stack([np.arange(6),[2,4,8,16,32,64]]));d.attrs['columns']='time_s,signal_mV';d.attrs['units']='s,mV'\n",
                str(project / "sim.h5"),
            ],
            check=True,
        )
    if "seismology" in case["servers"]:
        samples = [0.0, 1.0, 0.0, -1.0, 0.0]
        floats = [-12345.0] * 70
        floats[0], floats[5], floats[6] = 1.0, 0.0, 4.0
        ints = [-12345] * 40
        ints[6], ints[9] = 6, len(samples)
        (project / "waveform.sac").write_bytes(
            struct.pack("<70f", *floats)
            + struct.pack("<40i", *ints)
            + b" " * 192
            + struct.pack("<5f", *samples)
        )
    git(project, "init", "-q", "-b", "main")
    git(project, "config", "user.email", "eval@localhost")
    git(project, "config", "user.name", "Isolated evaluation")
    git(project, "add", ".")
    git(project, "commit", "-qm", "Research input fixture")
    (project / "research-notes.txt").write_text("Uncommitted: preserve this note.\n")
    protected = [
        "runs.csv",
        "gaps.csv",
        "research-notes.txt",
    ]
    if (project / "sim.h5").exists():
        protected.append("sim.h5")
    if (project / "waveform.sac").exists():
        protected.append("waveform.sac")
    return {
        name: hashlib.sha256((project / name).read_bytes()).hexdigest()
        for name in protected
    }


def check_artifacts(project: Path, case: dict, answer: str, protected: dict) -> dict:
    # A concise final message may link the substantive answer in an artifact.
    # Do not score supplied input notes or installed instructions as model output.
    for name in git(project, "ls-files", "--others", "--exclude-standard").splitlines():
        path = project / name
        if (
            path.suffix == ".md"
            and path.is_file()
            and not path.is_symlink()
            and not name.startswith((".agents/", ".codex/"))
            and path.stat().st_size <= 100000
        ):
            answer += "\n" + path.read_text(errors="replace")
    checks = {
        "source_preserved": all(
            (project / p).is_file()
            and hashlib.sha256((project / p).read_bytes()).hexdigest() == digest
            for p, digest in protected.items()
        )
    }
    checks.update(
        {
            "fact_" + str(i): bool(re.search(pattern, answer))
            for i, pattern in enumerate(case["facts"])
        }
    )
    if case["skill"] == "large-data-read":
        # Either a qualified sample or the independently known full reduction
        # is valid. Requiring the word "sample" would penalize an exact answer.
        checks = {
            key: value for key, value in checks.items() if not key.startswith("fact_")
        }
        numbers = [float(value) for value in re.findall(r"\b\d+\.\d+\b", answer)]
        full = any(
            math.isclose(value, 1 + 999 / 70000000, rel_tol=0, abs_tol=5e-11)
            for value in numbers
        )
        sampled = bool(
            re.search(r"(?i)sample|approximate", answer)
            and re.search(r"700[,.]?000|1(?:\.0+)?\s*%", answer)
            and any(value == 1 for value in numbers)
        )
        checks["qualified_mean_and_coverage"] = sampled or (
            full and bool(re.search(r"70[,.]?000[,.]?000", answer))
        )
    kind = case["kind"]
    if kind == "interpolate":
        try:
            values = [
                float(row["value"])
                for row in csv.DictReader((project / "cleaned.csv").open())
            ]
            checks["interpolation_values"] = values == [1, 2, 3, 4, 5]
        except (OSError, ValueError, KeyError):
            checks["interpolation_values"] = False
    if kind in {"plot", "visualization", "report"}:
        pngs = [p for p in project.rglob("*.png") if ".agents" not in p.parts]
        checks["png_created"] = any(
            p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in pngs
        )
    if kind == "slurm":
        text = (
            (project / "job.sbatch").read_text()
            if (project / "job.sbatch").exists()
            else ""
        )
        checks["resources"] = bool(
            re.search(r"(?:--ntasks(?:=|\s+)|-n\s*)16\b", text)
            and re.search(r"(?:01:30:00|90|1:30:00)", text)
            and "srun" in text
            and "./solver" in text
        )
        checks["no_invented_site"] = not re.search(
            r"#SBATCH.*(?:--account|--partition)", text
        )
    if kind == "report":
        helper = (
            ROOT
            / "plugins/clio-dataset-report/skills/dataset-report/scripts/verify_report.py"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(helper),
                "check",
                str(project / "report/clio-dataset-report.json"),
            ],
            capture_output=True,
            text=True,
        )
        checks["report_helper"] = result.returncode == 0
        try:
            rows = list(csv.DictReader((project / "report/data.csv").open()))
            checks["source_to_csv"] = [float(r["signal_mV"]) for r in rows] == [
                2,
                4,
                8,
                16,
                32,
                64,
            ] and [float(r["time_s"]) for r in rows] == list(range(6))
        except (OSError, KeyError, ValueError):
            checks["source_to_csv"] = False
    if kind == "bibliography":
        text = (
            (project / "references.bib").read_text()
            if (project / "references.bib").exists()
            else ""
        )
        checks["identifiers_present"] = (
            "2211.17192" in text and "2302.01318" in text and "@" in text
        )
    return checks
