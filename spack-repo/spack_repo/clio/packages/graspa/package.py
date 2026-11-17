# Copyright Spack Project Developers and IoWarp (clio-kit). SPDX-License-Identifier: BSD-3-Clause

from spack_repo.builtin.build_systems.generic import Package

from spack.package import *

SOURCES = ("axpy.cu", "main.cpp", "read_data.cpp", "data_struct.cpp", "VDW_Coulomb.cu")


class Graspa(Package):
    """gRASPA: GPU-accelerated Monte Carlo simulation of adsorption in nanoporous
    materials (zeolites, MOFs), from the Snurr group.

    +cuda builds nvc_main.x with the NVIDIA HPC SDK's nvc++ (as the project's
    NVC_COMPILE script does); +rocm builds hip_main.x with hipcc (HIP_COMPILE). One of
    the two is required: gRASPA has no CPU build."""

    homepage = "https://github.com/snurr-group/gRASPA"
    git = "https://github.com/snurr-group/gRASPA.git"

    maintainers("iowarp")
    license("MIT", checked_by="iowarp")

    version("2026.11.13", commit="c611ac2929d33640ac379825fdfb265017526b6d")

    variant("cuda", default=False, description="NVIDIA GPUs, built with nvc++ (NVHPC)")
    variant("rocm", default=False, description="AMD GPUs, built with hipcc")
    variant(
        "amdgpu_target",
        default="gfx90a",
        values=("gfx90a", "gfx940", "gfx942", "gfx1100"),
        description="AMD GPU architecture for +rocm",
    )
    conflicts("+cuda", when="+rocm", msg="choose one of +cuda and +rocm")
    conflicts("~cuda~rocm", msg="gRASPA needs a GPU build: add +cuda or +rocm")

    depends_on("cxx", type="build")
    depends_on("nvhpc", type="build", when="+cuda")
    depends_on("hip", when="+rocm")

    def install(self, spec, prefix):
        mkdirp(prefix.bin)
        with working_dir("src_clean"):
            if spec.satisfies("+cuda"):
                nvcxx = Executable(join_path(spec["nvhpc"].prefix, "Linux_x86_64",
                                             str(spec["nvhpc"].version), "compilers",
                                             "bin", "nvc++"))
                flags = ["-O3", "-std=c++20", "-target=gpu", "-Minline", "-fopenmp",
                         "-cuda", "-stdpar=multicore"]
                for src in SOURCES:
                    nvcxx(*flags, "-c", src)
                nvcxx(*flags, *[s.rsplit(".", 1)[0] + ".o" for s in SOURCES],
                      "-o", "nvc_main.x")
                install("nvc_main.x", prefix.bin)
            else:
                hipcc = Executable(join_path(spec["hip"].prefix.bin, "hipcc"))
                arch = f"--offload-arch={spec.variants['amdgpu_target'].value}"
                flags = ["-O3", "-std=c++20", arch, "-x", "hip", "-fgpu-rdc",
                         "-munsafe-fp-atomics", "-fopenmp", "-Wno-unused-result",
                         "-Wno-format"]
                for src in SOURCES:
                    hipcc(*flags, "-c", src)
                hipcc("-std=c++20", arch, "-fgpu-rdc", "-fopenmp", "--hip-link",
                      *[s.rsplit(".", 1)[0] + ".o" for s in SOURCES], "-o", "hip_main.x")
                install("hip_main.x", prefix.bin)
