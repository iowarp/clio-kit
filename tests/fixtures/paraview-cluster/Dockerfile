# The ParaView cluster fixture: three SSH hosts standing in for a site.
#   login   stands for `ares`, a login node: SSH from the person's machine,
#           reaches the package index, has no ParaView.
#   node08  stands for `ares-compute-08`: SSH only from login, ParaView 6
#           behind `module load paraview`, no route to the package index.
#   locked  a site that accepts only a password (interactive sign-in).
# Build every target with `cluster.sh build`; never by hand.
# Base images come from public.ecr.aws because Docker Hub rate-limits pulls.

ARG DEBIAN_IMAGE=public.ecr.aws/docker/library/debian:bookworm-slim

FROM ${DEBIAN_IMAGE} AS sshd-base
ARG CLUSTER_USER=researcher
ARG CLUSTER_UID=1000
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openssh-server ca-certificates curl procps iproute2 lmod \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid "${CLUSTER_UID}" --shell /bin/bash "${CLUSTER_USER}" \
 && mkdir -p /run/sshd /opt/modulefiles /etc/clio-pv \
 && rm -f /etc/ssh/ssh_host_*
COPY files/site-bashrc.sh /etc/clio-pv/site-bashrc.sh
# Site shell setup, as clusters do it: `module` works in every shell,
# including the non-interactive one `ssh host 'cmd'` runs (Debian's bash
# reads /etc/bash.bashrc for ssh commands before its interactive guard).
RUN { echo '. /etc/clio-pv/site-bashrc.sh'; cat /etc/bash.bashrc; } > /tmp/b \
 && mv /tmp/b /etc/bash.bashrc \
 && echo '/opt/modulefiles' > /etc/lmod/modulespath
COPY files/entrypoint.sh /usr/local/sbin/cluster-entrypoint
ENV CLUSTER_USER=${CLUSTER_USER}
EXPOSE 22
ENTRYPOINT ["/usr/local/sbin/cluster-entrypoint"]
CMD ["/usr/sbin/sshd", "-D", "-e"]

FROM sshd-base AS login
# A login node: a system Python with pip, no ParaView. Its Python (3.11)
# differs from node08's pvpython (3.12), as on real sites.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && ssh-keygen -A
COPY files/sshd-cluster.conf /etc/ssh/sshd_config.d/10-cluster.conf
LABEL org.iowarp.clio-pv-cluster.role=login

FROM sshd-base AS locked
ARG LOCKED_PASSWORD=locked-fixture-password
RUN echo "${CLUSTER_USER}:${LOCKED_PASSWORD}" | chpasswd && ssh-keygen -A
COPY files/sshd-locked.conf /etc/ssh/sshd_config.d/10-locked.conf
LABEL org.iowarp.clio-pv-cluster.role=locked

FROM sshd-base AS node08
ARG PARAVIEW_VERSION=6.2.1
ARG PYTHON_VERSION=3.12
ARG MICROMAMBA_VERSION=2.10.0-0
# Headless ParaView from conda-forge, installed where a site keeps its
# software. OSMesa (libosmesa6) is the software OpenGL it renders with.
# The extra CA secret is only for TLS-intercepting build networks; it is
# never written into a layer.
RUN --mount=type=secret,id=extra_ca,required=false \
    set -eu \
 && apt-get update \
 && apt-get install -y --no-install-recommends libosmesa6 bzip2 \
 && rm -rf /var/lib/apt/lists/* \
 && cat /etc/ssl/certs/ca-certificates.crt > /tmp/ca.pem \
 && if [ -s /run/secrets/extra_ca ]; then cat /run/secrets/extra_ca >> /tmp/ca.pem; fi \
 && curl -fsSL --cacert /tmp/ca.pem -o /tmp/micromamba \
      "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64" \
 && chmod +x /tmp/micromamba \
 && MAMBA_ROOT_PREFIX=/tmp/mamba /tmp/micromamba create -y --cacert-path /tmp/ca.pem \
      -p "/opt/apps/paraview/${PARAVIEW_VERSION}" -c conda-forge \
      "paraview=${PARAVIEW_VERSION}" "python=${PYTHON_VERSION}" pip \
 && rm -rf /tmp/mamba /tmp/micromamba /tmp/ca.pem \
 && cd "/opt/apps/paraview/${PARAVIEW_VERSION}" \
 && rm -rf include share/doc share/man pkgs conda-meta/*.json.bak \
 && find . -name '*.a' -delete \
 && ssh-keygen -A
COPY files/paraview.lua /opt/modulefiles/paraview/${PARAVIEW_VERSION}.lua
COPY files/make_datasets.py files/grow_series.py files/hold_dataset.py files/probe.py \
     files/start-pvserver.sh /opt/clio-pv-fixture/
RUN VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \
    "/opt/apps/paraview/${PARAVIEW_VERSION}/bin/python" \
      /opt/clio-pv-fixture/make_datasets.py /opt/datasets \
 && chmod 0755 /opt/clio-pv-fixture/start-pvserver.sh
COPY files/sshd-cluster.conf /etc/ssh/sshd_config.d/10-cluster.conf
LABEL org.iowarp.clio-pv-cluster.role=node08
