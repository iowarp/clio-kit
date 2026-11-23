"""The vtk.js fixes: each patched once, idempotent, and reported when it does not fit."""

from __future__ import annotations

from pathlib import Path

from paraview_mcp.node import vtkjs_patch

BRANCH = (
    "we(e.tmpMat4),e.tmpMat4[0]=2/v.usize,e.tmpMat4[12]=-1,e.tmpMat4[5]=2/v.vsize,"
    'e.tmpMat4[13]=-1,Ee(e.tmpMat4,e.tmpMat4,l),o.setUniformMatrix("MCPCMatrix",e.tmpMat4)'
)
HANDSHAKE = (
    "const X=s.getStillRatio()*Math.round(B+N),Y=s.getStillRatio()*Math.round(B+L);"
    "W===X&&H===Y?(F.unsubscribe(),u.getCanvasView().setBackgroundImage(k),P()):s.render()"
)
SYNC = "e.mtime=this.mtime;const a=this.renderWindow.synchronize(e);a&&(this.x=1)"


def test_the_display_branch_skips_a_null_inverse() -> None:
    text, outcome = vtkjs_patch.patch_text(f"a;{BRANCH};b")
    assert outcome == "patched"
    assert "e.tmpMat4[13]=-1,l&&Ee(e.tmpMat4,e.tmpMat4,l),o.setUniformMatrix" in text
    again, second = vtkjs_patch.patch_text(text)
    assert second == "already" and again == text


def test_a_remote_view_attaches_a_picture_rendered_for_another_size() -> None:
    text, outcome = vtkjs_patch.patch_text(
        f"a;{HANDSHAKE};b", vtkjs_patch.REMOTE_ATTACH
    )
    assert outcome == "patched"
    assert "s.render()" not in text
    assert (
        "(F.unsubscribe(),u.getCanvasView().setBackgroundImage(k),P(),(W!==X||H!==Y)&&setTimeout("
        in text
    )
    assert "clioCanvas.setSize(clioW+2,clioH+2),u.resize()" in text
    again, second = vtkjs_patch.patch_text(text, vtkjs_patch.REMOTE_ATTACH)
    assert second == "already" and again == text


def test_the_page_says_when_it_drew_a_scene_or_could_not() -> None:
    text, outcome = vtkjs_patch.patch_text(f"a;{SYNC};b", vtkjs_patch.SCENE_LOADED)
    assert outcome == "patched"
    assert "(this.renderWindow.synchronize(e));a&&(this.x=1)" in text
    assert text.count('new CustomEvent("vtk-scene-loaded"') == 3
    assert (
        "this.busy.stop(),this.renderWindow.getInteractor().setEnableRender(!0)" in text
    )
    again, second = vtkjs_patch.patch_text(text, vtkjs_patch.SCENE_LOADED)
    assert second == "already" and again == text


def test_a_bundle_without_the_branch_is_reported_not_changed(tmp_path: Path) -> None:
    assert vtkjs_patch.patch_text("nothing here")[1] == "branch_not_found"
    assert vtkjs_patch.apply(tmp_path) == "bundle_missing"
    bundle = tmp_path / vtkjs_patch.BUNDLE
    bundle.parent.mkdir(parents=True)
    bundle.write_text(BRANCH, encoding="utf-8")
    assert (
        vtkjs_patch.apply(tmp_path)
        == "display-actor guard: patched, remote attach: branch_not_found, "
        "scene loaded: branch_not_found"
    )
    assert (
        vtkjs_patch.apply(tmp_path)
        == "display-actor guard: already, remote attach: branch_not_found, "
        "scene loaded: branch_not_found"
    )
    bundle.write_text(f"{BRANCH};{HANDSHAKE};{SYNC}", encoding="utf-8")
    assert (
        vtkjs_patch.apply(tmp_path)
        == "display-actor guard: patched, remote attach: patched, scene loaded: patched"
    )
