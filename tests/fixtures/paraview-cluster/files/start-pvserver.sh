#!/bin/bash
# The node08 variant's ParaView: a pvserver the person started earlier on
# port 11111, with their own ParaView session attached and a dataset loaded.
# A pvserver keeps pipeline state only while a client is connected, and it
# exits when its first client leaves, so the person's session is modelled
# by a pvpython client that stays connected (hold_dataset.py). Further
# clients, such as the node half, see that session's data.
set -u
. /etc/clio-pv/site-bashrc.sh
module load paraview
port="${CLUSTER_PVSERVER_PORT:-11111}"
dataset="${CLUSTER_PVSERVER_DATASET:-/opt/datasets/pulse_series.pvd}"
logs=/tmp/clio-pv-fixture
mkdir -p "$logs"
pvserver --multi-clients --server-port="$port" --opengl-window-backend OSMesa \
    > "$logs/pvserver.log" 2>&1 &
for _ in $(seq 1 120); do
    ss -ltn "sport = :$port" | grep -q LISTEN && break
    sleep 0.5
done
pvpython /opt/clio-pv-fixture/hold_dataset.py --port "$port" --dataset "$dataset" \
    --ready "$logs/holder.ready" > "$logs/holder.log" 2>&1 &
wait
