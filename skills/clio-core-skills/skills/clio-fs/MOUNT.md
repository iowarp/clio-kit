# Mounting Context FS with clio_cte_fuse

`clio_cte_fuse` is a client of a running runtime. It binds the filesystem
pool by name (`clio_cte_filesystem`, or `CLIO_CFS_POOL`), learns the data
chain and stream pool from it, and serves the mount until unmounted.
Namespace calls go to the filesystem ChiMod; page reads and writes go
straight to the chain top through the CTE client; sizes and appends go to the
stream pool.

## Command

```
clio_cte_fuse <mountpoint> [FUSE options]
```

Every argument after the mountpoint goes to the FUSE library.

| option | effect |
|---|---|
| `-f` | stay in the foreground (recommended; the log goes to the terminal) |
| `-d` | trace every FUSE request |
| `-s` | single-threaded |
| `-o allow_other` | other users may open the mount (Linux: needs `user_allow_other` in `/etc/fuse.conf`) |
| `-o noatime`, `-o relatime`, `-o strictatime` | access-time policy; removed by the daemon before libfuse sees them and turned into `CLIO_FUSE_ATIME` |
| `-o backend=fskit` | macOS 15.4+ with macFUSE 5.1+: FSKit instead of the kernel extension; mountpoint must be under `/Volumes` |
| `-o uid=-1,gid=-1` | Windows: files belong to the mounting user (git's "dubious ownership" check needs it) |
| last argument `/dev/fd/N` | Linux/Apptainer `--fusemount`: drive a pre-opened FUSE fd at `CLIO_CTE_FUSE_MOUNTPOINT`. Needs a build against libfuse 3.14 headers; the manylinux wheels are built against 3.10 and print an error for this mode |

On Linux the daemon adds `-o default_permissions` (kernel checks mode bits)
and `-o max_threads=64` (libfuse 3.12+) unless the variables below say
otherwise. A Linux-only low-level variant, `clio_cte_fuse_ll`, exists in
source builds only (experimental, not installed).

## Platforms

### Linux (libfuse3; in the Linux wheel)

```bash
sudo apt install fuse3 libfuse3-3          # Fedora: sudo dnf install fuse3 fuse3-libs
clio-run start &
mkdir -p ~/clio-mnt
CLIO_WITH_RUNTIME=0 clio_cte_fuse ~/clio-mnt -f
# ... use it ...
fusermount3 -u ~/clio-mnt                  # not Ctrl+C: the daemon ignores it
clio-run stop
```

Docker: `docker run --device /dev/fuse --cap-add SYS_ADMIN ...`.
`fuse: device not found`: `sudo modprobe fuse`.

### Windows (WinFsp; in the Windows wheel)

Install WinFsp (winfsp.dev). The `clio_cte_fuse` console script prepends
`%ProgramFiles(x86)%\WinFsp\bin` to `PATH` and exits with "WinFsp not found"
if that folder is missing. Two PowerShell windows:

```powershell
# window 1
$env:CLIO_BIND_ADDR = "127.0.0.1"; $env:CLIO_SERVER_ADDR = "127.0.0.1"
clio-run start
# window 2
$env:CLIO_WITH_RUNTIME = "0"; $env:CLIO_SERVER_ADDR = "127.0.0.1"
clio_cte_fuse Z: -f -o uid=-1,gid=-1      # or an empty folder whose parent exists
# unmount
Stop-Process -Name clio_cte_fuse -Force
clio-run stop
```

No `fallocate` on Windows. `default_permissions` and `max_threads` are not
added. Microsoft Defender scans each new file over about 8 KB at close (about
10 ms each); a drive-letter path exclusion does not help, a process exclusion
(`Add-MpPreference -ExclusionProcess "git.exe"`) does (measured 90 to 1,800
creates/s for 22 KB files). Use `case_insensitive: true` on the filesystem
entry for NTFS-like name matching; it must be set before the volume is first
created.

### macOS (macFUSE; source build only)

The macOS wheel has no daemon binary. Build it:

```bash
brew install --cask macfuse                # approve the extension, reboot
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core && ./CI/ci-deps.sh --only-deps release && conda activate iowarp
cmake --preset release-mac-fuse -DCLIO_CORE_ENABLE_CONDA=ON
cmake --build build-mac-fuse -j"$(sysctl -n hw.ncpu)" && sudo cmake --install build-mac-fuse
CLIO_WITH_RUNTIME=0 clio_cte_fuse ~/clio-mnt -f    # FSKit: /Volumes/clio-mnt -o backend=fskit
umount ~/clio-mnt                          # or: diskutil unmount force ~/clio-mnt
```

If macFUSE is not found, the configure step skips the adapter with a warning
instead of failing: check the configure output. xattr `position` must be 0.

Other installs: Spack `spack install iowarp +fuse`; deb/rpm/AppImage and the
Windows installer include the daemon; the conda package does not.

## Environment variables of the daemon

"Multi-node" = the runtime's `networking.hostfile` (read from
`CLIO_SERVER_CONF`) has more than one line, or `CLIO_FUSE_MULTINODE=1`.
Times are seconds unless named `_MS`.

### Connecting

| variable | default | effect |
|---|---|---|
| `CLIO_WITH_RUNTIME` | unset = try to become the runtime, else attach | **Set `0`**: attach only. Unset with no runtime running, the daemon hosts a private runtime that nothing else sees. |
| `CLIO_WAIT_SERVER` | 600 s for the mount | how long to wait for the runtime (long enough for WAL replay after a reboot) |
| `CLIO_SERVER_ADDR`, `CLIO_PORT`, `CLIO_SERVER_CONF` | runtime defaults (port 9413) | same as every runtime client |
| `CLIO_CFS_POOL` | `clio_cte_filesystem` | filesystem pool name or `"major.minor"` |

### Caching and coherence

| variable | single-node default | multi-node default | effect |
|---|---|---|---|
| `CLIO_FUSE_MULTINODE` | from hostfile | from hostfile | `0`/`1` forces the mode |
| `CLIO_FUSE_ATTR_CACHE_S` | 1.0 | 0 | kernel attribute cache (also the negative-lookup cache). Use 0 on one node when several processes must see each other's metadata at once |
| `CLIO_FUSE_ENTRY_CACHE_S` | = attribute TTL | 1.0 | kernel name cache (safe: every getattr/open re-resolves at the owner) |
| `CLIO_FUSE_DIR_ATTR_CACHE_S` | = attribute TTL | 0 | daemon's cache of directory attributes; opt in (e.g. 1) on multi-node to stop one getattr per path component; another node's directory changes can then read stale that long |
| `CLIO_FUSE_WRITEBACK` | off | off | `1` = kernel writeback cache. Only for a single writer that fsyncs (corrupted git packs otherwise) |
| `CLIO_FUSE_STATFS_TTL_MS` | 1000 ms | 1000 ms | how long a capacity answer serves `statfs`; 0 = ask every call |

### Write, read and close path

| variable | default | effect |
|---|---|---|
| `CLIO_FUSE_APPEND` | `kernel` (single), `deferred` (multi) | `O_APPEND` offset: `kernel` trusts the kernel's cached size; `reserve` reserves the end at the stream home per write; `deferred` stages bytes for the home to merge (see STREAM_CHECKPOINT.md) |
| `CLIO_FUSE_ASYNC_FLUSH` | on (single-node only) | `close(2)` queues the size publish; operations that could observe the file wait for it. `0` awaits it |
| `CLIO_FUSE_ASYNC_CLOSE` | on | handle release is not awaited. `0` waits |
| `CLIO_FUSE_ASYNC_UTIMENS` | off | `1` detaches timestamp updates (single-node only); breaks `rsync -a`/`cp -a` ordering |
| `CLIO_FUSE_DIRECT_WRONLY` | on | write-only opens bypass the page cache (full-size writes instead of 4 KiB pieces). `0` disables |
| `CLIO_FUSE_ODIRECT` | off | `1` honours `O_DIRECT` (otherwise buffered so mmap keeps working) |
| `CLIO_FUSE_READAHEAD_PAGES` | 0 (off) | 1 MiB pages fetched ahead per sequential reader, max 64; needs `CLIO_FUSE_ATTR_CACHE_S` > 0; costs up to N+1 MiB per reading handle |
| `CLIO_FUSE_MAX_BACKGROUND` | libfuse default (12) | kernel read-ahead requests in flight; 256 suits parallel sequential reads |
| `CLIO_FUSE_MAX_THREADS` | 64 (Linux, libfuse 3.12+) | FUSE worker threads; 0 = libfuse default (10) |
| `CLIO_FUSE_READDIR_PREFETCH` | 0 (off) | for big listings, stat this many children concurrently and park results |
| `CLIO_FUSE_READDIR_PREFETCH_MIN` | 8192 entries | smallest listing that prefetches |
| `CLIO_FUSE_READDIR_PREFETCH_TTL_S` | 60 s | lifetime of a parked result |

### Attributes, permissions, diagnostics

| variable | default | effect |
|---|---|---|
| `CLIO_FUSE_PERMISSIONS` | on | `0` drops `-o default_permissions` (Linux). The ChiMod stores mode/owner but never enforces them |
| `CLIO_FUSE_ATIME` | `relatime` | `0`/`noatime`, `strict`/`strictatime` |
| `CLIO_FUSE_XATTR` | on | `0`: every xattr call returns ENOSYS (the kernel then stops sending them) |
| `CLIO_FUSE_STUCK_S` | 30 s | watchdog logs callbacks in flight this long (checked every 10 s); 0 disables |
| `CLIO_FUSE_PROFILE` | unset | report per-callback timings every N s (min 1 s) |
| `CLIO_ALLOW_PTRACE` | unset | `1` lets a debugger attach under restricted ptrace scope |
| `CLIO_CFS_NO_IMPLICIT_DIRS` | set to 1 by the daemon | skips the implicit-directory scan on negative lookups |
| `CTP_LOG_LEVEL`, `CTP_LOG_OUT` | runtime defaults | log level and file |

## Multi-node mounts

Run one runtime per node with the same compose file and a hostfile listing
every node, then one `clio_cte_fuse` per node with `CLIO_WITH_RUNTIME=0`.
All mounts see one namespace. Multi-node defaults (attribute TTL 0, deferred
appends, awaited close publish, no shared-memory attribute mirror) keep
close-to-open exact. Costs: each `stat` or lookup miss is a request; a `stat`
of a file homed elsewhere always asks its stream home for the size. Set
`CLIO_NET_DEAD_FAIL_FAST=1` on the runtimes to get EIO at once, instead of a
stall, for names a dead node owns.

## Unmounting

| platform | command |
|---|---|
| Linux | `fusermount3 -u <mnt>`; if busy or the runtime is gone, `fusermount3 -uz <mnt>` then kill the daemon |
| macOS | `umount <mnt>`; `diskutil unmount force <mnt>` |
| Windows | `Stop-Process -Name clio_cte_fuse -Force` |

Unmount before `clio-run stop`. Stopping the runtime under a live mount
leaves every syscall blocked (see SKILL.md Pitfalls).

## Other front ends (none are in the pip wheel)

| adapter | build option | how it is used | data home | `fsync` |
|---|---|---|---|---|
| POSIX interceptor `libclio_cte_posix.so` | `CLIO_CTE_ENABLE_POSIX_ADAPTER` (default ON) **and** `CLIO_CORE_ENABLE_ELF=ON` | `LD_PRELOAD`; only paths containing a `clio::` component are captured (`/scratch/clio::run1/out.dat` -> clio-fs `/scratch/run1/out.dat`); Linux/ELF only | clio-fs only, no backing file | synchronous writes; `fsync` makes the file durable like the mount |
| STDIO interceptor `libclio_cte_stdio.so` | `CLIO_CTE_ENABLE_STDIO_ADAPTER` + ELF | `LD_PRELOAD`; no user-space buffering; `fprintf`/`fscanf`/`setvbuf` not intercepted | clio-fs | as POSIX |
| MPI-IO `libclio_cte_mpiio.so` | `CLIO_CTE_ENABLE_MPIIO_ADAPTER` + ELF + `CLIO_CORE_ENABLE_MPI` | `LD_PRELOAD`; collectives run as independent I/O; file views ignored | clio-fs | `MPI_File_sync` is a no-op |
| HDF5 VFD `libclio_vfd.so` | `CLIO_CTE_ENABLE_VFD` | HDF5 plugin; native file stays authoritative, CTE is a byte-range cache (read tier off by default) | native file | native |
| HDF5 VOL `libclio_hdf5_vol.so` | `CLIO_CTE_ENABLE_HDF5_VOL` (HDF5 >= 1.14) | HDF5 plugin; caches dataset images | native file | native |
| ADIOS2 engine `iowarp_engine` | `CLIO_CTE_ENABLE_ADIOS2_ADAPTER` | ADIOS2 plugin engine; one blob per variable/step/rank; no variable metadata stored | CTE only | n/a |

`CLIO_CORE_ENABLE_HDF5` enables HDF5 in the CAE only; the VOL and VFD need
their own options. Preset `release-adapter` turns on MPI, HDF5 VOL, ADIOS2
and FUSE together.
