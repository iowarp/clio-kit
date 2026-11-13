# Site shell setup for the clio-pv cluster fixture, read first by
# /etc/bash.bashrc: Lmod's `module` in every bash shell, including the
# non-interactive one that `ssh host 'cmd'` runs.
if ! type module >/dev/null 2>&1; then
    . /etc/profile.d/lmod.sh
fi
