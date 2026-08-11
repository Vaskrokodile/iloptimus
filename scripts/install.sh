#!/bin/sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing IL Optimus…"
uv tool install --force "git+https://github.com/Vaskrokodile/iloptimus.git"

echo
echo "IL Optimus is installed. Starting the local app…"
exec iloptimus serve
