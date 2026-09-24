#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
TARGET="$REPOSITORY_ROOT/src/ss_nt_mpe_rc"
UPSTREAM='https://github.com/DeReKPIgg/ss-nt-mpe-rc.git'
COMMIT='abc0351cbd9bc999bfc52f5a8ff603f59d77473e'

test ! -e "$TARGET" || {
  echo "REFUSING: $TARGET already exists" >&2
  exit 2
}

git clone --no-checkout "$UPSTREAM" "$TARGET"
git -C "$TARGET" checkout --detach "$COMMIT"
test "$(git -C "$TARGET" rev-parse HEAD)" = "$COMMIT"
echo "UPSTREAM_INSTALL_COMPLETE commit=$COMMIT"
