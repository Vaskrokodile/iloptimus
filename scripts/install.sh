#!/bin/sh
set -eu

# Windows: redirect to the PowerShell installer
if [ "$(uname -s)" = "MINGW"* ] || [ "$(uname -s)" = "MSYS"* ] || [ "$(uname -s)" = "CYGWIN"* ]; then
  echo "Windows detected. Use the PowerShell installer instead:"
  echo "  powershell -ExecutionPolicy Bypass -File scripts/install.ps1"
  echo "Or download from:"
  echo "  https://raw.githubusercontent.com/Vaskrokodile/iloptimus/main/scripts/install.ps1"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing IL Optimus…"
uv tool install --force "git+https://github.com/Vaskrokodile/iloptimus.git"

echo
echo "IL Optimus is installed."
if [ "$(uname -s)" = "Darwin" ] && iloptimus install-desktop --force; then
  echo "Opening the native desktop app…"
  exec iloptimus desktop
fi

echo "Starting the local web app…"
exec iloptimus serve
