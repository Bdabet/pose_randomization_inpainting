#!/usr/bin/env bash
# Runs once when the devcontainer is first created. Just defers to the
# project's own install.sh so this stays in sync with it automatically --
# see .devcontainer/Dockerfile for the base toolchain this relies on.
set -euo pipefail

cd "$(dirname "$0")/.."

if git submodule status | grep -q '^-'; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

bash install.sh

if ! grep -q "conda activate phantom" ~/.bashrc; then
    echo "conda activate phantom" >> ~/.bashrc
fi

echo ""
echo "Setup complete. Open a new terminal (or run 'conda activate phantom') to use the phantom env."
