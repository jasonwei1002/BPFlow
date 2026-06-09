#!/usr/bin/env bash
# Force-sync local repo to remote (discards local commits & changes to TRACKED files).
# Untracked / gitignored files (rawdata/, wavflow/, plan/, CLAUDE.md ...) are kept.
# Usage: bash gitpull.sh          (sync current branch to remote main)
#        bash gitpull.sh <branch> (sync to a specific remote branch)
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="https://gh-proxy.org/https://github.com/jasonwei1002/BPFlow.git"
BRANCH="${1:-main}"

echo ">>> fetch ${BRANCH} <- ${REMOTE}"
git fetch "${REMOTE}" "${BRANCH}"

echo ">>> hard reset working tree to fetched remote state"
git reset --hard FETCH_HEAD

# To ALSO wipe untracked (non-ignored) files, uncomment — DESTRUCTIVE:
# git clean -fd

echo ">>> done. now at:"
git log --oneline -1
