#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_USER="${SUDO_USER:-$USER}"
VENV_DIR="$APP_DIR/.venv"
SERVICE_NAME="cpu-forecasting"

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<UNIT
[Unit]
Description=Cloud CPU Forecasting Streamlit Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${VENV_DIR}/bin/streamlit run ${APP_DIR}/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
sudo systemctl --no-pager --full status "${SERVICE_NAME}"

echo
echo "Deployment service installed. Open http://YOUR_EC2_PUBLIC_IP:8501 after allowing TCP 8501 in the EC2 security group."
echo "Logs: sudo journalctl -u ${SERVICE_NAME} -n 100 --no-pager"
