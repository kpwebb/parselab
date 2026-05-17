#!/usr/bin/env bash
# Pdfium binary fetcher for the Parselab Rust app. Downloads a release
# from github.com/bblanchon/pdfium-binaries and extracts it to
# vendor/pdfium/. Required by `pdfium-render` (and so by the desktop
# app's PDF rendering path).
#
# Usage:
#   ./vendor/setup-pdfium.sh                          # default build + auto platform
#   PDFIUM_BUILD=7811 ./vendor/setup-pdfium.sh        # pin build
#   PDFIUM_PLATFORM=mac-arm64 ./vendor/setup-pdfium.sh
#
# Pinned default is the chromium build originally validated.
set -euo pipefail

PDFIUM_BUILD="${PDFIUM_BUILD:-7811}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEST_DIR="${SCRIPT_DIR}/pdfium"

if [[ -n "${PDFIUM_PLATFORM:-}" ]]; then
    PLATFORM="${PDFIUM_PLATFORM}"
else
    case "$(uname -s)" in
        Darwin)
            case "$(uname -m)" in
                arm64)  PLATFORM="mac-arm64" ;;
                x86_64) PLATFORM="mac-x64" ;;
                *) echo "unknown macOS arch: $(uname -m)" >&2; exit 1 ;;
            esac
            ;;
        Linux)
            case "$(uname -m)" in
                x86_64)  PLATFORM="linux-x64" ;;
                aarch64) PLATFORM="linux-arm64" ;;
                *) echo "unknown Linux arch: $(uname -m)" >&2; exit 1 ;;
            esac
            ;;
        *)
            echo "unknown OS: $(uname -s)" >&2
            exit 1
            ;;
    esac
fi

URL="https://github.com/bblanchon/pdfium-binaries/releases/download/chromium%2F${PDFIUM_BUILD}/pdfium-${PLATFORM}.tgz"

echo "fetching pdfium build ${PDFIUM_BUILD} for ${PLATFORM}"
echo "  from ${URL}"
rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"
curl -fsSL "${URL}" | tar -xz -C "${DEST_DIR}"

echo
echo "installed:"
ls "${DEST_DIR}/lib" 2>/dev/null || ls "${DEST_DIR}"
