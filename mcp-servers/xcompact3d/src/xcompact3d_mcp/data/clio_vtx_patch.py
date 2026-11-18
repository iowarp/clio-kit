"""Make Xcompact3d v5.0's ADIOS2 snapshots readable by ParaView's ADIOS2VTXReader.

Upstream writes the global fields (ux, uy, uz, pp and the case fields) to
``data.bp5``, one ADIOS2 step per snapshot, but defines no ``vtk.xml``
attribute and no ``time`` variable. This local patch adds both and changes
nothing else:

* 2DECOMP&FFT 2.0.4 ``src/io.f90`` gains two public routines:
  ``decomp_2d_clio_attribute`` (a string attribute on an IO) and
  ``decomp_2d_clio_put_scalar`` (rank 0 puts one double in the open step).
* Incompact3d v5.0 ``src/visu.f90`` defines ``vtk.xml`` from ``clio-vtk.xml``
  in the run directory, when that file exists, before the engine opens. It puts
  the simulation time ``t`` as ``time`` in every snapshot step.

The clio-jarvis xcompact3d package writes ``clio-vtk.xml`` with the real grid.
Usage: ``clio_vtx_patch.py <2decomp-fft-src> <incompact3d-src>``. Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARK = "clio-jarvis VTX patch"

DECOMP_PUBLIC_OLD = "decomp_2d_start_io, decomp_2d_end_io, &\n             gen_iodir_name"
DECOMP_PUBLIC_NEW = (
    "decomp_2d_start_io, decomp_2d_end_io, &\n"
    "             decomp_2d_clio_attribute, decomp_2d_clio_put_scalar, &\n"
    "             gen_iodir_name"
)
DECOMP_ROUTINES = """
   ! MARK: metadata for ParaView's ADIOS2VTXReader. Not upstream.
   subroutine decomp_2d_clio_attribute(io_name, attr_name, attr_value)

      implicit none

      character(len=*), intent(in) :: io_name, attr_name, attr_value
#ifdef ADIOS2
      type(adios2_io) :: io
      type(adios2_attribute) :: attr
      integer :: ierror

      call adios2_at_io(io, adios, io_name, ierror)
      if (ierror /= 0) call decomp_2d_abort(__FILE__, __LINE__, ierror, "adios2_at_io")
      call adios2_define_attribute(attr, io, attr_name, attr_value, ierror)
      if (ierror /= 0) call decomp_2d_abort(__FILE__, __LINE__, ierror, "adios2_define_attribute")
#else
      associate (nm => io_name, an => attr_name, av => attr_value)
      end associate
#endif

   end subroutine decomp_2d_clio_attribute

   subroutine decomp_2d_clio_put_scalar(io_name, io_dir, var_name, val)

      implicit none

      character(len=*), intent(in) :: io_name, io_dir, var_name
      real(kind=8), intent(in) :: val
#ifdef ADIOS2
      type(adios2_io) :: io
      type(adios2_variable) :: var
      integer :: idx, ierror

      call adios2_at_io(io, adios, io_name, ierror)
      if (ierror /= 0) call decomp_2d_abort(__FILE__, __LINE__, ierror, "adios2_at_io")
      call adios2_inquire_variable(var, io, var_name, ierror)
      if (.not. var%valid) then
         call adios2_define_variable(var, io, var_name, adios2_type_dp, ierror)
         if (ierror /= 0) call decomp_2d_abort(__FILE__, __LINE__, ierror, "adios2_define_variable")
      end if
      idx = get_io_idx(io_name, io_dir)
      if (idx < 1) call decomp_2d_abort(__FILE__, __LINE__, idx, "clio: IO is not open")
      if (nrank == 0) then
         call adios2_put(engine_registry(idx), var, val, adios2_mode_sync, ierror)
         if (ierror /= 0) call decomp_2d_abort(__FILE__, __LINE__, ierror, "adios2_put")
      end if
#else
      associate (nm => io_name, dr => io_dir, vn => var_name, v => val)
      end associate
#endif

   end subroutine decomp_2d_clio_put_scalar

end module decomp_2d_io
""".replace("MARK", MARK)

VISU_READY_OLD = '    call decomp_2d_open_io(io_name, "data", mode)\n#endif'
VISU_READY_NEW = (
    "    ! " + MARK + "\n"
    "    if (mode == decomp_2d_write_mode) call clio_vtk_xml_attribute()\n"
    '    call decomp_2d_open_io(io_name, "data", mode)\n#endif'
)
VISU_START_OLD = '#ifdef ADIOS2\n    call decomp_2d_start_io(io_name, "data")\n#endif'
VISU_START_NEW = (
    '#ifdef ADIOS2\n    call decomp_2d_start_io(io_name, "data")\n'
    "    ! " + MARK + "\n"
    '    call decomp_2d_clio_put_scalar(io_name, "data", "time", real(t, kind=8))\n'
    "#endif"
)
VISU_USE_OLD = "    use decomp_2d_io, only : decomp_2d_start_io\n"
VISU_USE_NEW = (
    "    use decomp_2d_io, only : decomp_2d_start_io, decomp_2d_clio_put_scalar\n"
    "    use var, only : t\n"
)
VISU_HELPER_ANCHOR = "  subroutine visu_finalise()"
VISU_HELPER = """  ! MARK: define the vtk.xml attribute from ./clio-vtk.xml when present.
  subroutine clio_vtk_xml_attribute()

    use decomp_2d_io, only : decomp_2d_clio_attribute

    implicit none

    logical :: found
    integer :: unit, nbytes
    character(len=:), allocatable :: xml

    inquire(file="clio-vtk.xml", exist=found, size=nbytes)
    if (.not. found .or. nbytes <= 0) return
    allocate(character(len=nbytes) :: xml)
    open(newunit=unit, file="clio-vtk.xml", access="stream", form="unformatted", &
         action="read", status="old")
    read(unit) xml
    close(unit)
    call decomp_2d_clio_attribute(io_name, "vtk.xml", xml)
    if (nrank == 0) write(*, "(A,I0,A)") " clio: vtk.xml attribute defined (", nbytes, " bytes)"

  end subroutine clio_vtk_xml_attribute

""".replace("MARK", MARK)


def replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{where}: expected one anchor, found {count}: {old[:60]!r}")
    return text.replace(old, new)


def patch_decomp(src: Path) -> None:
    path = src / "src" / "io.f90"
    text = path.read_text()
    if MARK in text:
        print(f"{path}: already patched")
        return
    text = replace_once(text, DECOMP_PUBLIC_OLD, DECOMP_PUBLIC_NEW, str(path))
    text = replace_once(text, "\nend module decomp_2d_io", DECOMP_ROUTINES.rstrip("\n"), str(path))
    path.write_text(text if text.endswith("\n") else text + "\n")
    print(f"{path}: patched")


def patch_visu(src: Path) -> None:
    path = src / "src" / "visu.f90"
    text = path.read_text()
    if MARK in text:
        print(f"{path}: already patched")
        return
    text = replace_once(text, VISU_READY_OLD, VISU_READY_NEW, str(path))
    text = replace_once(text, VISU_START_OLD, VISU_START_NEW, str(path))
    text = replace_once(text, VISU_USE_OLD, VISU_USE_NEW, str(path))
    text = replace_once(text, VISU_HELPER_ANCHOR, VISU_HELPER + VISU_HELPER_ANCHOR, str(path))
    path.write_text(text)
    print(f"{path}: patched")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    patch_decomp(Path(argv[1]))
    patch_visu(Path(argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
