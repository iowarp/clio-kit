"""Let Xcompact3d v5.0's cylinder wake case immerse any CAD body.

Upstream ``itype = 5`` builds its immersed boundary epsilon field from an
analytic cylinder (``cex``, ``cey``, ``ra``). This local patch keeps that and
adds one path: when ``clio-geometry.txt`` exists in the run directory, the
epsilon field comes from a voxel mask of the body instead.

* ``clio-geometry.txt`` holds three lines: mask points ``nx ny nz``, origin
  ``x0 y0 z0`` and spacing ``dx dy dz`` in solver coordinates.
* ``clio-geometry.bin`` holds ``nx*ny*nz`` bytes, x fastest, 1 inside the body
  and 0 in the fluid.

Every solver point (and every refined point for ``iibm = 2``) takes the value
of its nearest mask voxel, so both IBM modes work. ``genepsi3d.f90`` passes
``dz`` through to the case so the lookup has a z coordinate. The case also
writes ``ep`` (epsilon) in each snapshot so viewers can draw the body.

Usage: ``clio_geometry_patch.py <incompact3d-src>``. Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARK = "clio geometry patch"

GENEPSI_OLD = "CALL geomcomplex_cyl(epsi, nxi, nxf, ny, nyi, nyf, nzi, nzf, dx, yp, remp)"
GENEPSI_NEW = (
    "CALL geomcomplex_cyl(epsi, nxi, nxf, ny, nyi, nyf, nzi, nzf, dx, yp, dz, remp) ! " + MARK
)

CYL_STATE_OLD = "  integer :: FS\n"
CYL_STATE_NEW = """  integer :: FS
  ! MARK: voxel mask state
  logical, save :: clio_geom_checked = .false., clio_geom_on = .false.
  integer, save :: cg_n(3)
  real(mytype), save :: cg_o(3), cg_d(3)
  integer(kind=1), allocatable, save :: cg_mask(:,:,:)
""".replace("MARK", MARK)

CYL_SIG_OLD = "  subroutine geomcomplex_cyl(epsi,nxi,nxf,ny,nyi,nyf,nzi,nzf,dx,yp,remp)"
CYL_SIG_NEW = "  subroutine geomcomplex_cyl(epsi,nxi,nxf,ny,nyi,nyf,nzi,nzf,dx,yp,dz,remp)"

CYL_DECL_OLD = "    real(mytype)               :: dx\n    real(mytype)               :: remp\n"
CYL_DECL_NEW = (
    "    real(mytype)               :: dx, dz\n"
    "    real(mytype)               :: remp\n"
    "    integer                    :: ii, jj, kk\n"
)

CYL_BODY_OLD = "    ! Intitialise epsi\n    epsi(:,:,:)=zero\n"
CYL_BODY_NEW = """    ! Intitialise epsi
    epsi(:,:,:)=zero

    ! MARK: body from a voxel mask when clio-geometry.txt is present
    if (.not. clio_geom_checked) call clio_geom_load()
    if (clio_geom_on) then
       do k=nzi,nzf
          kk = nint((real(k-1,mytype)*dz - cg_o(3))/cg_d(3)) + 1
          if (kk < 1 .or. kk > cg_n(3)) cycle
          do j=nyi,nyf
             jj = nint((yp(j) - cg_o(2))/cg_d(2)) + 1
             if (jj < 1 .or. jj > cg_n(2)) cycle
             do i=nxi,nxf
                ii = nint((real(i-1,mytype)*dx - cg_o(1))/cg_d(1)) + 1
                if (ii < 1 .or. ii > cg_n(1)) cycle
                if (cg_mask(ii,jj,kk) /= 0) epsi(i,j,k)=remp
             enddo
          enddo
       enddo
       return
    endif
""".replace("MARK", MARK)

CYL_LOAD_ANCHOR = "  end subroutine geomcomplex_cyl\n"
CYL_LOAD = """  end subroutine geomcomplex_cyl

  ! MARK: read clio-geometry.txt and clio-geometry.bin once
  subroutine clio_geom_load()

    implicit none

    logical :: found
    integer :: unit

    clio_geom_checked = .true.
    inquire(file="clio-geometry.txt", exist=found)
    if (.not. found) return
    open(newunit=unit, file="clio-geometry.txt", status="old", action="read")
    read(unit,*) cg_n
    read(unit,*) cg_o
    read(unit,*) cg_d
    close(unit)
    allocate(cg_mask(cg_n(1),cg_n(2),cg_n(3)))
    open(newunit=unit, file="clio-geometry.bin", access="stream", form="unformatted", &
         status="old", action="read")
    read(unit) cg_mask
    close(unit)
    clio_geom_on = .true.
    if (nrank == 0) write(*,"(A,3(1X,I0),A,I0)") " clio: geometry mask", cg_n, &
         ", solid voxels ", count(cg_mask /= 0)

  end subroutine clio_geom_load
""".replace("MARK", MARK)

VISU_INIT_OLD = '    call decomp_2d_register_variable(io_name, "critq", 1, 0, output2D, mytype)\n'
VISU_INIT_NEW = (
    VISU_INIT_OLD
    + "    ! "
    + MARK
    + "\n"
    + '    call decomp_2d_register_variable(io_name, "ep", 1, 0, output2D, mytype)\n'
)

VISU_WRITE_OLD = (
    '    call write_field(di1, ".", "critq", num, flush = .true.) '
    "! Reusing temporary array, force flush\n"
)
VISU_WRITE_NEW = (
    VISU_WRITE_OLD
    + "    ! "
    + MARK
    + "\n"
    + '    call write_field(ep1, ".", "ep", num, .true., flush = .true.)\n'
)


def replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{where}: expected one anchor, found {count}: {old[:60]!r}")
    return text.replace(old, new)


def patch(src: Path) -> None:
    genepsi = src / "src" / "genepsi3d.f90"
    text = genepsi.read_text()
    if MARK not in text:
        genepsi.write_text(replace_once(text, GENEPSI_OLD, GENEPSI_NEW, str(genepsi)))
        print(f"{genepsi}: patched")
    else:
        print(f"{genepsi}: already patched")

    case = src / "src" / "Case-Cylinder-wake.f90"
    text = case.read_text()
    if MARK in text:
        print(f"{case}: already patched")
        return
    for old, new in (
        (CYL_STATE_OLD, CYL_STATE_NEW),
        (CYL_SIG_OLD, CYL_SIG_NEW),
        (CYL_DECL_OLD, CYL_DECL_NEW),
        (CYL_BODY_OLD, CYL_BODY_NEW),
        (CYL_LOAD_ANCHOR, CYL_LOAD),
        (VISU_INIT_OLD, VISU_INIT_NEW),
        (VISU_WRITE_OLD, VISU_WRITE_NEW),
    ):
        text = replace_once(text, old, new, str(case))
    case.write_text(text)
    print(f"{case}: patched")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    patch(Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
