#!/bin/bash
# Copy fixtures into an eval run's empty workspace before the agent starts.
# Usage (from a case's setup.sh): place.sh SOURCE:DESTINATION ...
# A DESTINATION starting with ~/ goes under the run's temporary home.
set -euo pipefail
fixtures=$(cd "$(dirname "$0")" && pwd)
for pair in "$@"; do
    source=${pair%%:*}
    destination=${pair#*:}
    # shellcheck disable=SC2088  # a literal "~/" prefix is the convention here
    case $destination in "~/"*) destination=$HOME/${destination#"~/"} ;; esac
    mkdir -p "$(dirname "$destination")"
    cp -R "$fixtures/$source" "$destination"
done
