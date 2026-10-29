#!/usr/bin/env bash
# Run from the repository root (the folder that contains neurosight/ and README.md).
# These files at the root are copies of files that live inside neurosight/crates/*:
#   main.rs, ui.rs  -> identical to crates/app/src/main.rs, ui.rs
#   lib.rs          -> identical to crates/perception/src/lib.rs
#   pose.rs         -> OLDER copy of crates/events/src/pose.rs (no head/reach analyzers)
#   neurosight.zip  -> archive of the workspace
# Check `git log` / `git diff` first if you are unsure, then run this.
set -euo pipefail
git rm -q main.rs ui.rs lib.rs pose.rs neurosight.zip
# The old neurosight/README.md is superseded by the root README.md.
git rm -q neurosight/README.md
echo "Done. The CAD files stay at the repo root, as the README describes."
