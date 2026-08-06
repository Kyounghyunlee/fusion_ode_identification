# Local workstation environment (Fedora, RTX 5080). Source from the repo root:
#   source scripts/env_local.sh
# Replaces the ITER SDCC module-load wrapper for this machine.

_CERT=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export SSL_CERT_FILE="$_CERT"
export REQUESTS_CA_BUNDLE="$_CERT"
export AWS_CA_BUNDLE="$_CERT"
export CURL_CA_BUNDLE="$_CERT"
export UV_NATIVE_TLS=1

export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=1

# shellcheck disable=SC1091
source .venv/bin/activate
