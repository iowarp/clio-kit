"""Groupby never silently replaces existing research output or the input."""

from pandas_mcp.implementation.transformations import groupby_operations


def test_groupby_output_requires_explicit_overwrite(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("group,value\na,1\na,3\n")
    original = source.read_bytes()
    output = tmp_path / "input_grouped.csv"
    output.write_text("existing research output\n")
    arguments = dict(
        file_path=str(source), group_by=["group"], operations={"value": "mean"}
    )
    result = groupby_operations(**arguments)
    assert not result["success"] and result["error_type"] == "FileExistsError"
    assert output.read_text() == "existing research output\n"
    alternate = tmp_path / "means.csv"
    result = groupby_operations(**arguments, output_file=str(alternate))
    assert result["success"] and result["results"] == [{"group": "a", "value": 2.0}]
    assert output.read_text() == "existing research output\n"
    result = groupby_operations(**arguments, overwrite=True)
    assert result["success"] and output.read_bytes() == alternate.read_bytes()
    for destination in (source, tmp_path / "input-link.csv"):
        if destination != source:
            destination.hardlink_to(source)
        result = groupby_operations(
            **arguments, output_file=str(destination), overwrite=True
        )
        assert not result["success"]
        assert source.read_bytes() == original
