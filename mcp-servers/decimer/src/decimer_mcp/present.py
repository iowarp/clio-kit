"""The agent-facing text of each tool: lean, in words."""

from __future__ import annotations

from typing import Any


def smiles_text(data: dict[str, Any]) -> str:
    image, rdkit, timing = data["image"], data["rdkit"], data["timing"]
    kind = "hand-drawn" if data["hand_drawn"] else "printed"
    lines = [
        (
            f"DECIMER ({kind}-structure model) read {image['path']} "
            f"({image['width']} x {image['height']} px {image['format'].upper()}) as SMILES: "
            f"{data['smiles']}"
        )
    ]
    if rdkit["valid"]:
        lines.append(
            f"RDKit parses it: canonical {rdkit['canonical_smiles']}, formula "
            f"{rdkit['formula']}, {rdkit['molecular_weight_g_mol']} g/mol, "
            f"{rdkit['heavy_atoms']} heavy atoms, {rdkit['stereo_centres']} stereo centre(s)"
            + (f", InChIKey {rdkit['inchikey']}." if rdkit.get("inchikey") else ".")
        )
    else:
        lines.append(f"RDKit cannot use it: {rdkit['error']}. The raw string is kept above.")
    conf = data["confidence"]
    if conf["mean"] is not None:
        lines.append(
            f"Token confidence: mean {conf['mean']:.3f}, lowest {conf['min']:.3f} "
            "(per-token, not a probability that the structure is right)."
        )
    if timing["downloaded_mb"]:
        lines.append(
            f"First use: downloaded the {', '.join(timing['downloaded'])} model weights "
            f"(about {timing['downloaded_mb']} MB from Zenodo) into {data['model_cache']}; "
            f"downloading and loading took {timing['load_s']:.0f} s. Later calls reuse them."
        )
    elif timing["cold_start"]:
        lines.append(f"Loaded the model in {timing['load_s']:.0f} s (cached from now on).")
    lines.append(f"Reading took {timing['predict_s']:.1f} s.")
    lines.append(data["verify"])
    return "\n".join(lines)


def crop_text(data: dict[str, Any]) -> str:
    image = data["image"]
    left, top, right, bottom = data["box_pixels"]
    return (
        f"Cropped pixels left {left}, top {top}, right {right}, bottom {bottom} of "
        f"{data['source']} into {image['path']} ({image['width']} x {image['height']} px "
        "PNG). Pass that path to image_to_smiles."
    )
