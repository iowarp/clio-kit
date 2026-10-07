# ChronoLog Installation and Deployment Guide

This guide walks you through steps 0 to 7 for installing and deploying ChronoLog. ([github.com](https://github.com/grc-iit/ChronoLog/wiki/Tutorial-1%3A-First-Steps-with-ChronoLog))

## Step 0: Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Git** – Required for cloning the ChronoLog repository. Install with:

  ```bash
  sudo apt install git  # Debian/Ubuntu
  ```
* **g++** – The GNU C++ compiler is needed to build ChronoLog. Install with:

  ```bash
  sudo apt install g++  # Debian/Ubuntu
  ```

* **cmake** – The cmake is needed to build ChronoLog. Install with:

  ```bash
  sudo apt install cmake  # Debian/Ubuntu
  ```


Once these are installed, you're ready for the next step.

## Step 1: Clone the Repository

Open your terminal in the desired directory and run:

```bash
git clone -b develop https://github.com/grc-iit/ChronoLog.git
```

This command downloads the ChronoLog code to your local machine.

## Step 2: Install the Spack Package Manager

Spack simplifies managing software builds. To install Spack:

```bash
git clone --branch v0.21.2 https://github.com/spack/spack.git
source /path-to-where-spack-was-cloned/spack/share/spack/setup-env.sh
```

Replace `/path-to-where-spack-was-cloned/` with your actual path.

## Step 3: Move to the Deploy Folder

Navigate to ChronoLog's deploy directory:

```bash
cd /path-to-repo/deploy
```

Ensure `/path-to-repo/` is the location where you cloned ChronoLog.

## Step 4: Build the Project

Build ChronoLog in Debug mode:

```bash
./local_single_user_deploy.sh -b -t Debug
```

This compiles the project and may take a few minutes.

## Step 5: Install the Project

Install ChronoLog locally:

```bash
./local_single_user_deploy.sh -i -w /home/$USER/chronolog/Debug
```

This sets up ChronoLog in `/home/$USER/chronolog/Debug`.

## Step 6: Deploy the Project

Deploy ChronoLog:

```bash
./local_single_user_deploy.sh -d -w /home/$USER/chronolog/Debug
```

ChronoLog is now running and ready for use.

## Step 7: Check the Installation

Verify the deployment by checking running processes:

```bash
pgrep -laf "chronovisor_server|chrono_grapher|chrono_keeper|chrono_player"
```

You should see four processes corresponding to ChronoLog components.


Configure the MCP process with the interpreter matching the native extension:

```bash
export UV_PYTHON=3.11  # use the ABI of your py_chronolog_client build
export LD_LIBRARY_PATH="$HOME/chronolog/Debug/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/chronolog/Debug/lib:${PYTHONPATH:-}"
```

Use the actual build/install paths on your machine. Do not rename or symlink an
extension to disguise a Python ABI mismatch. Keep these variables in the
ChronoLog server's agent configuration, separate from other native servers.

## Step 8: Launch the Interactive Client Admin

Let’s fire up the interactive client! Open your terminal, navigate to the /bin directory on the location where ChronoLog is installed, and run:

```bash
./client_admin -i -c ../conf/visor_conf.json
```

Feel free to test and explore the chronolog operations.

### Reader Script

Build the reader directly from this checkout against the same ChronoLog source
and libraries used by the services. Follow the configurable CMake commands in
[Native client and archive reader](../README.md#native-client-and-archive-reader).
Set `HDF5_READER_BIN` to the installed executable and `CHRONO_CONF` to your
service configuration. Test it with:

```bash
"$HDF5_READER_BIN" -c "$CHRONO_CONF" -C YOUR_CHRONICLE -S YOUR_STORY
```

After recording and stopping a session, allow the native archive flush interval
before checking retrieval. Successful acceptance means recovering the exact
recorded text, including quotes/newlines, rather than merely finding processes
or connecting to the MCP server.
