#!/usr/bin/env bash
set -euo pipefail

RUN=${RUN:-results/upstream_randopt_official_p32/countdown_20260507_135251}
OUT=${OUT:-$RUN/upstream_smoke_audit}
PYTHON=${PYTHON:-python}
MIN_POPULATION=${MIN_POPULATION:-1}
MIN_TEST_SAMPLES=${MIN_TEST_SAMPLES:-1}
REQUIRE_PAPER_SCALE=${REQUIRE_PAPER_SCALE:-0}

args=(
  --run "$RUN"
  --out "$OUT"
  --min-population "$MIN_POPULATION"
  --min-test-samples "$MIN_TEST_SAMPLES"
)

if [[ "$REQUIRE_PAPER_SCALE" == "1" ]]; then
  args+=(--require-paper-scale)
fi

"$PYTHON" -m randopt_lora_lab.upstream_smoke_audit "${args[@]}"
echo "upstream smoke audit written to $OUT"
