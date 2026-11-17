"""Run DECIMER on an image and check its SMILES with RDKit.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), tools/ocsr_backends.py and
tools/ocsr_core.py: DECIMER's ``predict_SMILES`` is loaded once and cached, the first
call reports that it paid the cold start, and the model's output is treated as
untrusted text (length-capped) before RDKit parses it.

TensorFlow is imported only inside this server's own process, the first time an image
is read. DECIMER keeps its weights through ``pystow``; this module points pystow at a
folder in the user cache (``platformdirs``) so the two ~300 MB Zenodo archives
(printed and hand-drawn models) are downloaded once and reused.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TypedDict

from platformdirs import user_cache_dir

from .errors import DecimerError


# What DECIMER downloads (zip sizes on Zenodo, checked 2026-11). Reported to the agent
# on the call that downloads them, so a long first call is explained.
class _Model(TypedDict):
    name: str
    folder: str
    zip_mb: int
    url: str


MODELS: dict[bool, _Model] = {
    False: {
        "name": "DECIMER",
        "folder": "DECIMER_model",
        "zip_mb": 299,
        "url": "https://zenodo.org/record/8300489/files/models.zip",
    },
    True: {
        "name": "DECIMER_HandDrawn",
        "folder": "DECIMER_HandDrawn_model",
        "zip_mb": 299,
        "url": "https://zenodo.org/records/10781330/files/DECIMER_HandDrawn_model.zip",
    },
}
MAX_SMILES_CHARS = 4000
CACHE_ENV = "DECIMER_MCP_CACHE"

_lock = threading.Lock()
_loaded: set[bool] = set()


def cache_dir() -> Path:
    """Where the model weights live: ``$DECIMER_MCP_CACHE`` or the user cache folder."""
    return Path(os.environ.get(CACHE_ENV) or user_cache_dir("decimer-mcp", "iowarp"))


def _model_present(hand_drawn: bool) -> bool:
    folder = cache_dir() / "DECIMER-V2" / MODELS[hand_drawn]["folder"]
    return (folder / "saved_model.pb").is_file()


def _prepare_env() -> None:
    os.environ["PYSTOW_HOME"] = str(cache_dir())
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def _predict(
    image_path: str, hand_drawn: bool
) -> tuple[str, list[tuple[str, float]], dict[str, Any]]:
    """Return (smiles, token confidences, timing) for one image, loading the model once."""
    _prepare_env()
    started = time.monotonic()
    with _lock:
        cold = hand_drawn not in _loaded
        # The tokenizer ships in the printed model's archive, so the hand-drawn model's
        # first use downloads both archives.
        missing = [flag for flag in (False, True) if not _model_present(flag)]
        # DECIMER and pystow print progress to stdout, which is the MCP stdio channel.
        with contextlib.redirect_stdout(sys.stderr):
            try:
                import DECIMER
                from DECIMER import decimer as decimer_module

                if cold:
                    decimer_module.load_models(printed=not hand_drawn, hand_drawn=hand_drawn)
                    _loaded.add(hand_drawn)
            except (OSError, ConnectionError) as exc:
                raise DecimerError(
                    "model_download_failed",
                    f"The DECIMER model could not be downloaded or opened ({exc}). It comes "
                    f"from Zenodo ({MODELS[hand_drawn]['url']}, about "
                    f"{MODELS[hand_drawn]['zip_mb']} MB) into {cache_dir()}; check the "
                    "network and the free disk space, then try again.",
                    cache_dir=str(cache_dir()),
                ) from None
            loaded_at = time.monotonic()
            try:
                smiles, tokens = DECIMER.predict_SMILES(
                    image_path, confidence=True, hand_drawn=hand_drawn
                )
            except Exception as exc:  # noqa: BLE001 - any model failure is reported as is
                raise DecimerError(
                    "model_failed",
                    f"DECIMER could not read {image_path}: {type(exc).__name__}: {exc}",
                ) from None
    done = time.monotonic()
    downloaded = [MODELS[flag] for flag in missing if _model_present(flag)]
    timing = {
        "cold_start": cold,
        "load_s": round(loaded_at - started, 2) if cold else 0.0,
        "predict_s": round(done - loaded_at, 2),
        "downloaded": [m["name"] for m in downloaded],
        "downloaded_mb": sum(m["zip_mb"] for m in downloaded),
    }
    return smiles, [(str(t), float(c)) for t, c in tokens], timing


def check_smiles(smiles: str) -> dict[str, Any]:
    """RDKit's view of a SMILES: valid or not, canonical form, formula, mass, InChIKey."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    text = (smiles or "").strip()
    if not text:
        return {"valid": False, "error": "the model returned no SMILES"}
    if len(text) > MAX_SMILES_CHARS:
        return {"valid": False, "error": f"longer than {MAX_SMILES_CHARS} characters"}
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return {"valid": False, "error": "RDKit could not parse it (bad valence or syntax)"}
    try:
        inchikey = Chem.MolToInchiKey(mol) or None
    except Exception:  # noqa: BLE001 - InChI is optional in some RDKit builds
        inchikey = None
    return {
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight_g_mol": round(Descriptors.MolWt(mol), 3),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "stereo_centres": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "inchikey": inchikey,
    }


def image_to_smiles(image: dict[str, Any], hand_drawn: bool) -> dict[str, Any]:
    """Read ``image`` (from :func:`images.check_image`) with DECIMER and check the result."""
    smiles, tokens, timing = _predict(image["path"], hand_drawn)
    confidences = [c for _, c in tokens]
    rdkit = check_smiles(smiles)
    return {
        "image": image,
        "model": MODELS[hand_drawn]["name"],
        "hand_drawn": hand_drawn,
        "smiles": smiles,
        "rdkit": rdkit,
        "confidence": {
            "mean": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "min": round(min(confidences), 4) if confidences else None,
            "lowest_tokens": [
                {"token": t, "confidence": round(c, 4)}
                for t, c in sorted(tokens, key=lambda tc: tc[1])[:5]
            ],
            "meaning": "the decoder's per-token probability, not a probability that the "
            "whole structure is right",
        },
        "timing": timing,
        "model_cache": str(cache_dir()),
        "verify": "This SMILES is a model's reading of the drawing. Show it to the person "
        "(e.g. open it in Ketcher) and have them compare it with the image before using it.",
    }
