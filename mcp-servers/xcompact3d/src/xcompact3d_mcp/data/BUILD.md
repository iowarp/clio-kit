# Build Xcompact3d for xcompact3d-mcp

xcompact3d-mcp prepares cases for Xcompact3d (Incompact3d) v5.0, built with the
ADIOS2 I/O backend and two small local patches, both shipped with this server:

- `clio_vtx_patch.py` makes each snapshot readable by ParaView's
  `ADIOS2VTXReader` while the run is still writing (a `vtk.xml` attribute and
  the simulation time in every step). It patches 2DECOMP&FFT 2.0.4 and
  Incompact3d's `visu.f90`.
- `clio_geometry_patch.py` lets the cylinder wake case (`itype = 5`) take its
  body from a voxel mask (`clio-geometry.txt` and `clio-geometry.bin`) and
  writes the body (`ep`) in every snapshot.

Both are idempotent and touch nothing else. Run the build on a compute node
of the machine that will run the solver.

## Needs

- An MPI Fortran compiler (`mpif90`/`mpifort`, GNU 11 or later), CMake 3.20 or
  later and ADIOS2 2.9 or later built with MPI and Fortran bindings. On a
  machine without them, one conda environment holds all three:
  `micromamba create -p $ENV -c conda-forge openmpi-mpifort adios2=*=mpi_openmpi* cmake make`.
- The two release tarballs:
  - 2DECOMP&FFT 2.0.4: `https://github.com/2decomp-fft/2decomp-fft/archive/refs/tags/v2.0.4.tar.gz`
  - Incompact3d 5.0: `https://github.com/xcompact3d/Incompact3d/archive/refs/tags/v5.0.tar.gz`

## Steps

```bash
PREFIX=~/.clio/xcompact3d/v5.0-clio        # anything; 'current' points at it
PATCHES=<the folder build_instructions names>
mkdir -p ~/src && cd ~/src
tar xf 2decomp-fft-2.0.4.tar.gz && tar xf Incompact3d-5.0.tar.gz
python3 $PATCHES/clio_vtx_patch.py 2decomp-fft-2.0.4 Incompact3d-5.0
python3 $PATCHES/clio_geometry_patch.py Incompact3d-5.0

# 2DECOMP&FFT with ADIOS2 I/O
cmake -S 2decomp-fft-2.0.4 -B build-2decomp -DCMAKE_BUILD_TYPE=Release \
  -DIO_BACKEND=adios2 -DDOUBLE_PRECISION=ON -Dadios2_DIR=$ENV/lib/cmake/adios2 \
  -DCMAKE_INSTALL_PREFIX=$PREFIX/2decomp
cmake --build build-2decomp -j 16 && cmake --install build-2decomp

# Incompact3d against it
cmake -S Incompact3d-5.0 -B build-x3d -DCMAKE_BUILD_TYPE=Release \
  -DIO_BACKEND=adios2 -DDOUBLE_PRECISION=ON \
  -Ddecomp2d_DIR=$PREFIX/2decomp/lib/decomp2d -DCMAKE_INSTALL_PREFIX=$PREFIX
cmake --build build-x3d -j 16 && cmake --install build-x3d

ln -sfn $PREFIX ~/.clio/xcompact3d/current
```

Then write `$PREFIX/bin/xcompact3d-env.sh`, which every run script sources
before `mpirun`. It puts the MPI and ADIOS2 libraries in reach, for example:

```bash
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$PREFIX/2decomp/lib:${LD_LIBRARY_PATH:-}
# Single-node runs on clusters whose fabric OpenMPI cannot use:
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm
```

`prepare_case` uses `~/.clio/xcompact3d/current/bin/xcompact3d` unless the
`executable` argument or `XCOMPACT3D_BIN` names another.

## Check

A run directory from `prepare_case`, run with `bash run.sh`, must print
`clio: geometry mask nx ny nz, solid voxels N` near the top of
`xcompact3d.log`, grow `data.bp5` by one step every `ioutput` time steps and
append a line to `forces.dat` every time step. `check_case` reads all three.
