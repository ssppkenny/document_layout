#!/usr/bin/env bash
# Update the ocr-reflow-lib submodule to the latest upstream commit and record the bump.
set -euo pipefail

SUBMODULE_PATH="src/ocr_reflow/ocr_reflow_lib"

cd "$(dirname "$0")"

if [ ! -f "$SUBMODULE_PATH/.git" ]; then
    echo "ERROR: submodule not initialized at $SUBMODULE_PATH" >&2
    exit 1
fi

git -C "$SUBMODULE_PATH" fetch origin

if ! git -C "$SUBMODULE_PATH" diff --quiet; then
    echo "ERROR: submodule has uncommitted changes; commit or stash them first" >&2
    exit 1
fi

NEW_SHA="$(git -C "$SUBMODULE_PATH" rev-parse origin/main)"
git -C "$SUBMODULE_PATH" checkout --quiet "$NEW_SHA"

if git diff --quiet -- "$SUBMODULE_PATH"; then
    echo "Already up to date at $NEW_SHA"
    exit 0
fi

git add "$SUBMODULE_PATH"
SHORT="$(git -C "$SUBMODULE_PATH" log -1 --format=%h)"
git commit -m "Update ocr-reflow-lib to $SHORT"
echo "Bumped $SUBMODULE_PATH to $SHORT"
