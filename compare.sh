#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <dir1> <dir2>" >&2
  exit 1
fi

d1="${1%/}"
d2="${2%/}"

while IFS= read -r -d '' f1; do
  rel="${f1#"$d1"/}"                
  f2="$d2/$rel"
  if [[ -f "$f2" ]]; then
    if diff -q --strip-trailing-cr "$f1" "$f2" >/dev/null; then
      echo "SAME  : $rel"
    else
      echo "DIFF  : $rel"
    fi
  else
    echo "ONLY1 : $rel"
  fi
done < <(find "$d1" -type f -name '*.py' -print0)

while IFS= read -r -d '' f2; do
  rel="${f2#"$d2"/}"
  f1="$d1/$rel"
  [[ -f "$f1" ]] || echo "ONLY2 : $rel"
done < <(find "$d2" -type f -name '*.py' -print0)

