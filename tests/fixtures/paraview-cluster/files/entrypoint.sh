#!/bin/sh
# Entry point of every cluster container: site trust and egress settings,
# home ownership, the optional pvserver of the node08 variant, then sshd.
set -eu
user="${CLUSTER_USER:-researcher}"
: > /etc/environment

# An extra CA for TLS-intercepting networks (mounted by `cluster.sh up` on
# login only). It joins the system trust store that curl and pip read.
if [ -s /etc/clio-pv/extra-ca.crt ]; then
    cp /etc/clio-pv/extra-ca.crt /usr/local/share/ca-certificates/clio-pv-extra.crt
    update-ca-certificates >/dev/null 2>&1 \
        || echo "cluster-entrypoint: update-ca-certificates failed" >&2
    bundle=/etc/ssl/certs/ca-certificates.crt
    printf 'SSL_CERT_FILE=%s\nREQUESTS_CA_BUNDLE=%s\nPIP_CERT=%s\n' \
        "$bundle" "$bundle" "$bundle" >> /etc/environment
fi
# An egress proxy, when the network needs one (pam_env gives it to SSH
# sessions through /etc/environment).
if [ -n "${CLUSTER_HTTPS_PROXY:-}" ]; then
    printf 'HTTPS_PROXY=%s\nhttps_proxy=%s\nNO_PROXY=%s\nno_proxy=%s\n' \
        "$CLUSTER_HTTPS_PROXY" "$CLUSTER_HTTPS_PROXY" \
        "${CLUSTER_NO_PROXY:-}" "${CLUSTER_NO_PROXY:-}" >> /etc/environment
fi

chown "$user:" "/home/$user"
chmod 0750 "/home/$user"

if [ "${CLUSTER_START_PVSERVER:-0}" = "1" ]; then
    runuser -u "$user" -- /opt/clio-pv-fixture/start-pvserver.sh &
fi

exec "$@"
