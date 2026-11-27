---
title: Decimer MCP
description: "Read a chemical structure drawing (PNG, JPEG) into SMILES with the DECIMER image-to-SMILES model, checked with RDKit."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Decimer"
  icon="🖼️"
  category="Utilities"
  description="Read a chemical structure drawing (PNG, JPEG) into SMILES with the DECIMER image-to-SMILES model, checked with RDKit."
  version="1.0.0"
  actions={["image_to_smiles", "crop_image"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "chemistry", "OCSR", "DECIMER", "SMILES", "RDKit", "image"]}
  license="BSD-3-Clause"
  tools={[{"name": "image_to_smiles", "description": "Read the chemical structure drawn in an image (PNG, JPEG, GIF, TIFF, BMP, WEBP; one molecule) into a SMILES string with DECIMER, then check it with RDKit: valid or not, canonical SMILES, formula, molecular weight (g/mol), stereo centres, InChIKey, and the model's per-token confidence. An invalid SMILES is still returned as the model wrote it. hand_drawn=true uses DECIMER's model for sketches. The result is a model's reading and must be checked against the image.", "function_name": "image_to_smiles"}, {"name": "crop_image", "description": "Cut a rectangle out of an image (for example one structure from a paper's figure) and save it as PNG under output_dir/chemistry/decimer/. box is [left, top, right, bottom] in pixels from the top-left corner, or as fractions of the width and height with unit='fraction'. Returns the new image's path.", "function_name": "crop_image"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `image_to_smiles`, `crop_image`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
