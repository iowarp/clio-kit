# CLIO Spack repository

Spack packages for programs CLIO Kit's servers run but never download or build
themselves. Namespace `clio`, Spack 1.0 repository layout (`spack_repo/clio/`).

| Package | Builds | Used by |
| --- | --- | --- |
| `fdmnes` | FDMNES (Institut Néel, CNRS) from its Fortran source: serial, Gaussian solver, LAPACK/BLAS. Installs `bin/fdmnes`. | the `fdmnes` server (`FDMNES_COMMAND`) |
| `graspa` | gRASPA (Snurr group, MIT) for NVIDIA GPUs with nvc++ (`+cuda`, `bin/nvc_main.x`) or AMD GPUs with hipcc (`+rocm`, `bin/hip_main.x`). There is no CPU build. | the `graspa` server (`GRASPA_COMMAND`) |

## Confined use

Everything Spack writes for CLIO goes under one folder, `$CLIO_SPACK_ROOT` (for example
`<workspace>/.spack`): the install tree, build stages, source and misc caches, and
Spack's user configuration and cache. Nothing is written to `~/.spack`.

```sh
export CLIO_SPACK_ROOT=$PWD/.spack
export CLIO_KIT_SPACK_REPO=/path/to/clio-kit/spack-repo
export SPACK_USER_CONFIG_PATH=$CLIO_SPACK_ROOT/user-config
export SPACK_USER_CACHE_PATH=$CLIO_SPACK_ROOT/user-cache
export SPACK_DISABLE_LOCAL_CONFIG=1
spack -C $CLIO_KIT_SPACK_REPO/config repo list      # clio, then builtin
spack -C $CLIO_KIT_SPACK_REPO/config compiler find
spack -C $CLIO_KIT_SPACK_REPO/config install fdmnes
export FDMNES_COMMAND=$(spack -C $CLIO_KIT_SPACK_REPO/config location -i fdmnes)/bin/fdmnes
```

`config/config.yaml` sets the install tree, stages and caches under `$CLIO_SPACK_ROOT`;
`config/repos.yaml` adds this repository (through `$CLIO_KIT_SPACK_REPO`) ahead of
Spack's builtin packages. On a GPU node, `spack ... install graspa +cuda` (or `+rocm
amdgpu_target=gfx942`), then `GRASPA_COMMAND=$(spack ... location -i graspa)/bin/nvc_main.x`.

## What was checked

With Spack 1.0.4 on Linux x86-64 (Ubuntu 24.04, gcc 13.3), all state confined as above:

- `fdmnes`: `spack install fdmnes` fetched the source archive, built OpenBLAS and FDMNES
  (36 s for FDMNES), and the fdmnes server's integration tests passed with the
  installed `bin/fdmnes` (Cu K edge; FeO6 octahedron).
- `graspa`: `spack audit packages` passes, and `graspa +cuda` (nvhpc 25.5) and
  `graspa +rocm` (hip 6.4.1) concretize. **Not built**: no GPU and no NVHPC or ROCm
  toolchain here.

FDMNES publishes its source behind one share link that always holds the newest
revision, so the `fdmnes` checksum fails when FDMNES releases a new one. Update the
version (the revision date) and sha256 then.
