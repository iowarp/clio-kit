"""Input checks and cleanup shared by the visualization engine and the server.

Kept out of ``paraview_capabilities.py`` and ``server.py`` so both stay within
their file-size ratchet. ``paraview`` is imported inside the functions that need
it, so this module loads without a ParaView installation.
"""

import os

IMPORT_HELP = (
    "ParaView's Python modules could not be imported by this server. Launch it "
    "with UV_PYTHON set to the Python version your ParaView was built with, "
    "PYTHONPATH set to that same ParaView's site-packages (a stale or mismatched "
    "PYTHONPATH inherited from the shell causes this), and LD_LIBRARY_PATH set to "
    "its shared libraries."
)
CONNECT_FAILED = (
    "Could not connect to ParaView server at {host}:{port}. Start pvserver on "
    "that port (it exits when its client disconnects, so start it again for a new "
    "session) and pass --server HOST --pv-port PORT after `--` if it is not the "
    "default."
)


def discard_temporary(previous_source, temporary_filter):
    """Restore the active source and remove a filter created only to compute.

    A read-only analysis must not leave its temporary filter active (later tools
    would act on it) or in the pipeline.
    """
    from paraview.simple import Delete, SetActiveSource

    if temporary_filter is not None:
        SetActiveSource(previous_source)
        Delete(temporary_filter)


def stl_name_problem(stl_filename, data_directory):
    """Return an error unless ``stl_filename`` stays inside the data directory."""
    if os.path.basename(stl_filename) == stl_filename and stl_filename.strip("."):
        return None
    return (
        f"Error: stl_filename must be a plain file name, not a path: "
        f"'{stl_filename}'. It is saved in {data_directory}."
    )


def contour_for(existing, source, field, value):
    """Return ``(contour, problem)`` for contouring ``source`` at ``value``.

    Reuses ``existing`` or creates the Contour filter. An isovalue outside the
    field's range (or an unknown field) yields an empty surface, so it is refused
    with the range instead: a contour created here is removed again and the
    previously active source restored, leaving the pipeline unchanged.
    """
    from paraview.simple import Contour, GetActiveSource

    previous_source = GetActiveSource()
    contour = existing or Contour(Input=source)
    name = field or contour.ContourBy[1]
    array = source.GetDataInformation().GetPointDataInformation()
    array = array.GetArrayInformation(name)
    if not array:
        problem = f"field '{name}' is not a point-data array of the source"
    else:
        low, high = array.GetComponentRange(0)
        if low <= value <= high:
            return contour, None
        problem = (
            f"isovalue {value} is outside the data range [{low}, {high}] "
            f"of field '{name}'"
        )
    if not existing:
        discard_temporary(previous_source, contour)
    return None, f"Error: {problem}. Isosurface not changed."


def representation_problem(display, rep_type):
    """Return an error for a type the representation does not offer.

    ParaView silently ignores an unknown representation type.
    """
    available = list(display.GetProperty("Representation").Available)
    if rep_type in available:
        return None
    return (
        f"Error: Unknown representation type '{rep_type}'. "
        f"Available types: {', '.join(available)}"
    )
