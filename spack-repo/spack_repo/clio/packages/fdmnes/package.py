# Copyright Spack Project Developers and IoWarp (clio-kit). SPDX-License-Identifier: BSD-3-Clause

from spack_repo.builtin.build_systems.makefile import MakefilePackage

from spack.package import *

MPIF_SERIAL = """\
! Serial build of FDMNES: the MPI constants the stub routines in not_mpi.f90 use.
      integer MPI_COMM_WORLD, MPI_INTEGER, MPI_REAL8, MPI_LOGICAL
      integer MPI_INTEGER8, MPI_DOUBLE_PRECISION, MPI_DOUBLE_COMPLEX
      integer MPI_STATUS_SIZE, MPI_IN_PLACE
      parameter (MPI_COMM_WORLD=0, MPI_INTEGER=1, MPI_REAL8=2, MPI_LOGICAL=3)
      parameter (MPI_INTEGER8=4, MPI_DOUBLE_PRECISION=5, MPI_DOUBLE_COMPLEX=6)
      parameter (MPI_STATUS_SIZE=6, MPI_IN_PLACE=-1)
"""

OBJECTS = (
    "main clemf0 coabs convolution diffraction dirac fdm fdmx fprime fprime_data general "
    "lecture mat metric minim optic potential selec scf spgroup sphere tab_data tddft "
    "tensor tools ATA_subroutines RIXS mat_solve_gaussian not_mpi"
)


class Fdmnes(MakefilePackage):
    """FDMNES: ab initio simulation of X-ray absorption (XANES), resonant X-ray
    diffraction and related spectroscopies, by finite differences or multiple
    scattering (Institut Neel, CNRS).

    This recipe builds the serial program from the Fortran source with the Gaussian
    solver and LAPACK (the README's build without MUMPS). FDMNES publishes its source
    as one archive behind a fixed share link that always holds the newest revision, so
    the checksum below fails when FDMNES releases a new revision: update the version
    (the revision date in Doc/FDMNES_Modifications.txt of the binary package) and the
    sha256 then."""

    homepage = "https://fdmnes.neel.cnrs.fr/"
    # The share token contains digits Spack would read as a version, so each version
    # names its URL explicitly instead of a class-level url.
    source_share = "https://hub.neel.cnrs.fr/index.php/s/Lpg2GD3A2H4PETZ/download"

    maintainers("iowarp")
    license("UNKNOWN", checked_by="iowarp")  # free download; no licence file in the source

    version(
        "2026.11.05",
        sha256="049fe6d62bfb07965d99560c096b9039d7d3a85518c376cfabd88e7dc7026230",
        url="https://hub.neel.cnrs.fr/index.php/s/Lpg2GD3A2H4PETZ/download",
        expand=True,
        extension="zip",
    )

    depends_on("fortran", type="build")
    depends_on("lapack")
    depends_on("blas")

    # Spack unpacks the archive's one folder (fdmnes_fortran_prog) as the stage root.
    build_directory = "."

    def edit(self, spec, prefix):
        with working_dir(self.build_directory):
            with open("mpif.h", "w") as fh:
                fh.write(MPIF_SERIAL)
            libs = (spec["lapack"].libs + spec["blas"].libs).ld_flags
            flags = "-O2 -w -I."
            if spec.satisfies("%gcc"):
                flags += " -fallow-argument-mismatch -fallow-invalid-boz"
            objs = " ".join(f"{name}.o" for name in OBJECTS.split())
            with open("Makefile.spack", "w") as fh:
                fh.write(
                    f"FC = {spack_fc}\nFFLAGS = {flags}\nOBJ = {objs}\n\n"
                    "all: fdmnes\n\n"
                    "fdmnes: $(OBJ)\n"
                    f"\t$(FC) -o $@ $(OBJ) {libs}\n\n"
                    # main.f90 defines the module every other file uses.
                    "$(filter-out main.o,$(OBJ)): main.o\n\n"
                    "%.o: %.f90\n\t$(FC) $(FFLAGS) -c $< -o $@\n"
                )

    @property
    def build_targets(self):
        return ["-f", "Makefile.spack", "all"]

    def install(self, spec, prefix):
        mkdirp(prefix.bin)
        install(join_path(self.build_directory, "fdmnes"), prefix.bin)
