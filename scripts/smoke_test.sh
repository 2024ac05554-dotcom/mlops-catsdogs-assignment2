#!/usr/bin/env bash
# Post-deploy / post-build smoke test.
# Calls the health endpoint and one prediction call; exits non-zero (failing
# the pipeline) if either check fails.
#
# Usage: scripts/smoke_test.sh http://localhost:8000 [path/to/sample.jpg]

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
SAMPLE_IMAGE="${2:-}"
MAX_RETRIES=15
SLEEP_SECONDS=2

echo "==> Smoke testing $BASE_URL"

# --- Wait for the service to become healthy ---
attempt=0
until curl -fsS "$BASE_URL/health" > /tmp/health_response.json; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "FAIL: /health did not become available after $MAX_RETRIES attempts"
    exit 1
  fi
  echo "Waiting for service to be healthy (attempt $attempt/$MAX_RETRIES)..."
  sleep "$SLEEP_SECONDS"
done

echo "Health check response:"
cat /tmp/health_response.json
echo

if ! grep -q '"status":"ok"' /tmp/health_response.json && ! grep -q '"status": "ok"' /tmp/health_response.json; then
  echo "FAIL: health endpoint did not report status=ok"
  exit 1
fi
echo "PASS: /health OK"

# --- Prediction call ---
if [ -z "$SAMPLE_IMAGE" ]; then
  # Generate a throwaway 100x100 JPEG on the fly if no sample was provided
  SAMPLE_IMAGE="/tmp/smoke_test_sample.jpg"
  python3 - "$SAMPLE_IMAGE" <<'EOF'
import sys
from PIL import Image
Image.new("RGB", (100, 100), color=(128, 64, 32)).save(sys.argv[1])
EOF
fi

echo "==> Calling /predict with $SAMPLE_IMAGE"
PREDICT_RESPONSE=$(curl -fsS -X POST "$BASE_URL/predict" -F "file=@${SAMPLE_IMAGE};type=image/jpeg")
echo "Predict response: $PREDICT_RESPONSE"

if ! echo "$PREDICT_RESPONSE" | grep -q '"label"'; then
  echo "FAIL: /predict did not return a label"
  exit 1
fi
echo "PASS: /predict OK"

echo "==> Smoke tests passed"
exit 0
