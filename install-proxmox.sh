#!/usr/bin/env bash
# Advanced OCR LLM — Proxmox LXC Installer
#
# Run on the Proxmox HOST shell:
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/Twozee-Tech/Advanced-OCR-LLM/main/install-proxmox.sh)"
#
# Creates a Debian 12 LXC, installs Docker, builds and runs the OCR container.
# The OCR engine itself runs on a separate Ollama host — no GPU required here.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "  ${RED}✗${NC} $*"; }
info() { echo -e "  ${BLUE}→${NC} $*"; }

ask() {
    local reply
    printf "%b" "$1" > /dev/tty
    read -r reply < /dev/tty
    echo "${reply:-$2}"
}
ask_secret() {
    local reply
    printf "%b" "$1" > /dev/tty
    read -rs reply < /dev/tty
    echo "" > /dev/tty
    echo "$reply"
}

REPO="Twozee-Tech/Advanced-OCR-LLM"
APP_DIR="/opt/ocr"

# ── banner ────────────────────────────────────────────────────────────────────
echo -e "${BLUE}${BOLD}"
echo "╔══════════════════════════════════════════════╗"
echo "║   Advanced OCR LLM                          ║"
echo "║   Proxmox LXC + Docker Installer            ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  PDF → Ollama (qwen3-vl) → Markdown"
echo "  Web UI + Marker-compatible API (/convert)"
echo "  Idle RAM: ~150 MB  |  Disk: ~1.5 GB"
echo ""

# ── 1. verify Proxmox host ────────────────────────────────────────────────────
echo -e "${YELLOW}[1/6] Checking Proxmox host...${NC}"
if ! command -v pct &>/dev/null; then
    err "pct not found — run this on the Proxmox VE host shell."
    exit 1
fi
ok "Proxmox VE detected"

HOST_ARCH=$(uname -m)
case "$HOST_ARCH" in
    aarch64) ARCH="arm64" ;;
    x86_64)  ARCH="amd64" ;;
    *) err "Unsupported architecture: $HOST_ARCH"; exit 1 ;;
esac
ok "Host architecture: $ARCH"

# ── 2. find / download Debian 12 template ────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/6] Finding Debian 12 LXC template...${NC}"

TEMPLATE_FILE=$(find /var/lib/vz/template/cache/ -name "debian-12-*${ARCH}*.tar.*" 2>/dev/null | sort -V | tail -1 || true)

if [[ -z "$TEMPLATE_FILE" ]]; then
    info "Not found locally — checking Proxmox mirrors..."
    pveam update 2>/dev/null || true
    TEMPLATE_NAME=$(pveam available --section system 2>/dev/null \
        | awk '{print $2}' | grep -E "debian-12.*${ARCH}" | sort -V | tail -1 || true)

    if [[ -n "$TEMPLATE_NAME" ]]; then
        pveam download local "$TEMPLATE_NAME"
        TEMPLATE_FILE="/var/lib/vz/template/cache/$TEMPLATE_NAME"
    else
        warn "Proxmox mirrors have no ${ARCH} template. Downloading from linuxcontainers.org..."
        LC_BASE="https://images.linuxcontainers.org/images/debian/bookworm/${ARCH}/default"
        LC_VER=$(curl -s "${LC_BASE}/" | grep -oP '\d{8}_\d+:\d+' | tail -1)
        if [[ -z "$LC_VER" ]]; then
            err "Could not fetch template list from linuxcontainers.org"
            exit 1
        fi
        TEMPLATE_FILE="/var/lib/vz/template/cache/debian-12-standard_${ARCH}.tar.xz"
        info "Downloading ${LC_VER} rootfs (~100 MB)..."
        wget -q --show-progress \
            "${LC_BASE}/${LC_VER}/rootfs.tar.xz" \
            -O "$TEMPLATE_FILE"
    fi
fi

ok "Template: $(basename "$TEMPLATE_FILE")"
TEMPLATE_STOR="local:vztmpl/$(basename "$TEMPLATE_FILE")"

# ── 3. LXC configuration ──────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/6] Configure LXC...${NC}"
echo ""

echo "  Available storage:"
pvesm status --content rootdir 2>/dev/null \
    | awk 'NR>1 {printf "    %-20s %s GiB free\n", $1, int($5/1024/1024)}' || true
echo ""

DEFAULT_STORAGE=$(pvesm status --content rootdir 2>/dev/null \
    | awk 'NR>1 {print $1; exit}')
DEFAULT_STORAGE=${DEFAULT_STORAGE:-local-lvm}

CTID=$(pvesh get /cluster/nextid 2>/dev/null || echo "200")
CTID=$(ask     "  Container ID [${CTID}]: "            "$CTID")
STORAGE=$(ask  "  Storage [${DEFAULT_STORAGE}]: "      "$DEFAULT_STORAGE")
HOSTNAME=$(ask "  Hostname [ocr-pipeline]: "            "ocr-pipeline")
RAM=$(ask      "  RAM MB [2048]: "                     "2048")
DISK=$(ask     "  Disk GB [8]: "                       "8")
CORES=$(ask    "  CPU cores [2]: "                     "2")

echo ""
LAST_IP=$(grep -h '^net0:' /etc/pve/lxc/*.conf 2>/dev/null \
    | grep -oE 'ip=[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' \
    | grep -oE '[0-9]+$' | sort -n | tail -1)
SUGGEST_LAST=$(( ${LAST_IP:-199} + 1 ))
DEFAULT_IP="192.168.0.${SUGGEST_LAST}/24"

CT_IP=$(ask  "  Container IP [${DEFAULT_IP}]: "  "$DEFAULT_IP")
GW=$(ask     "  Gateway [192.168.0.1]: "          "192.168.0.1")
NET_IP="ip=${CT_IP},gw=${GW}"

# ── 4. application configuration ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/6] Configure application...${NC}"
echo ""

echo -e "  ${BOLD}Ollama${NC} — the host running qwen3-vl:30b"
OLLAMA_URL=$(ask   "  Ollama URL [http://192.168.0.169:11434]: "  "http://192.168.0.169:11434")
OLLAMA_MODEL=$(ask "  Model [qwen3-vl:30b]: "                     "qwen3-vl:30b")
OLLAMA_TEMP=$(ask  "  Temperature [0.25]: "                       "0.25")
OLLAMA_CTX=$(ask   "  Context tokens [30000]: "                   "30000")

echo ""
echo -e "  ${BOLD}Web UI${NC}"
WEB_USER=$(ask     "  Username [admin]: "   "admin")
WEB_PASS=$(ask_secret "  Password: ")
[[ -z "$WEB_PASS" ]] && { err "Password cannot be empty."; exit 1; }
WEB_PORT=$(ask     "  Port [14000]: "       "14000")
DATA_PATH=$(ask    "  Data path [/data/ocr]: " "/data/ocr")

# ── 5. create LXC ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/6] Creating LXC ${CTID}...${NC}"

BRIDGE=$(grep -h 'bridge=' /etc/pve/lxc/*.conf 2>/dev/null \
    | grep -oE 'bridge=[^,]+' | cut -d= -f2 \
    | sort | uniq -c | sort -rn | awk 'NR==1{print $2}')
BRIDGE=${BRIDGE:-vmbr0}
info "Using network bridge: ${BRIDGE}"

pct create "$CTID" "$TEMPLATE_STOR" \
    --hostname "$HOSTNAME" \
    --memory   "$RAM" \
    --cores    "$CORES" \
    --rootfs   "${STORAGE}:${DISK}" \
    --net0     "name=eth0,bridge=${BRIDGE},${NET_IP}" \
    --unprivileged 1 \
    --features nesting=1 \
    --ostype   debian \
    --start    1 \
    --onboot   1

ok "Container ${CTID} created and started"
info "Waiting for boot..."
sleep 6

# ── 6. provision: Docker + build + run ───────────────────────────────────────
echo ""
echo -e "${YELLOW}[6/6] Provisioning (Docker, clone, build, deploy)...${NC}"
info "This takes 2–4 minutes on first run."

pct exec "$CTID" -- bash -euo pipefail << PROVISION
export DEBIAN_FRONTEND=noninteractive

echo "--- Updating packages ---"
apt-get update -qq
apt-get install -y --no-install-recommends \
    curl git ca-certificates gnupg lsb-release

echo "--- Installing Docker ---"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian \$(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable docker
systemctl start docker
echo "Docker: \$(docker --version)"

echo "--- Cloning repository ---"
git clone --depth=1 "https://github.com/${REPO}.git" "${APP_DIR}"

echo "--- Building Docker image ---"
cd "${APP_DIR}"
docker build -f docker/Dockerfile -t ocr-pipeline .
echo "Image built."
PROVISION

# Pass credentials via pct exec env (avoids embedding in heredoc)
pct exec "$CTID" -- bash -euo pipefail -c "
mkdir -p '${DATA_PATH}'
docker run -d \
  --name ocr-pipeline \
  --restart unless-stopped \
  -p ${WEB_PORT}:14000 \
  -v '${DATA_PATH}:/data' \
  -e OCR_WEB_MODE=true \
  -e OCR_WEB_USER='${WEB_USER}' \
  -e OCR_WEB_PASS='${WEB_PASS}' \
  -e OLLAMA_URL='${OLLAMA_URL}' \
  -e OLLAMA_MODEL='${OLLAMA_MODEL}' \
  -e OLLAMA_TEMPERATURE='${OLLAMA_TEMP}' \
  -e OLLAMA_NUM_CTX='${OLLAMA_CTX}' \
  ocr-pipeline
"

ok "Container deployed"

CONTAINER_IP=$(pct exec "$CTID" -- hostname -I 2>/dev/null | awk '{print $1}' || echo "<container-ip>")

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════╗"
echo "║   Installation complete!                     ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Web UI:    ${BOLD}http://${CONTAINER_IP}:${WEB_PORT}${NC}"
echo -e "  Login:     ${WEB_USER} / (your password)"
echo -e "  Ollama:    ${OLLAMA_URL}  (${OLLAMA_MODEL})"
echo ""
echo "  Obsidian plugin → Self-hosted (Docker):"
echo -e "    URL: ${BOLD}http://${CONTAINER_IP}:${WEB_PORT}${NC}"
echo ""
echo "  Management (run on Proxmox host):"
echo "    pct exec ${CTID} -- docker logs -f ocr-pipeline"
echo "    pct exec ${CTID} -- docker restart ocr-pipeline"
echo ""
echo "  Update:"
echo "    pct exec ${CTID} -- bash -c 'cd ${APP_DIR} && git pull && docker build -f docker/Dockerfile -t ocr-pipeline . && docker restart ocr-pipeline'"
echo ""
