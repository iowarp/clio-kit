"""The vtk.js display-actor guard: patched once, idempotent, and reported when it does not fit."""

from __future__ import annotations

from pathlib import Path

from paraview_mcp.node import vtkjs_patch

BRANCH = (
    "we(e.tmpMat4),e.tmpMat4[0]=2/v.usize,e.tmpMat4[12]=-1,e.tmpMat4[5]=2/v.vsize,"
    'e.tmpMat4[13]=-1,Ee(e.tmpMat4,e.tmpMat4,l),o.setUniformMatrix("MCPCMatrix",e.tmpMat4)'
)


def test_the_display_branch_skips_a_null_inverse() -> None:
    text, outcome = vtkjs_patch.patch_text(f"a;{BRANCH};b")
    assert outcome == "patched"
    assert "e.tmpMat4[13]=-1,l&&Ee(e.tmpMat4,e.tmpMat4,l),o.setUniformMatrix" in text
    again, second = vtkjs_patch.patch_text(text)
    assert second == "already" and again == text


def test_a_bundle_without_the_branch_is_reported_not_changed(tmp_path: Path) -> None:
    assert vtkjs_patch.patch_text("nothing here")[1] == "branch_not_found"
    assert vtkjs_patch.apply(tmp_path) == "bundle_missing"
    bundle = tmp_path / vtkjs_patch.BUNDLE
    bundle.parent.mkdir(parents=True)
    bundle.write_text(BRANCH, encoding="utf-8")
    assert vtkjs_patch.apply(tmp_path) == "patched"
    assert vtkjs_patch.apply(tmp_path) == "already"
