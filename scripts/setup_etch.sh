#!/usr/bin/env bash
# =============================================================================
#  Memory Etch — Quick Setup
#  =============================================================================
#  Usage:
#    ./scripts/setup_etch.sh              Check environment
#    ./scripts/setup_etch.sh --serve      Start viewer on port 9120
#    ./scripts/setup_etch.sh --install    Install dependencies
#    ./scripts/setup_etch.sh --db PATH    Specify DB path
# =============================================================================
set -euo pipefail

MINT='\033[38;2;112;255;214m'
DIM='\033[38;2;102;102;102m'
BOLD='\033[1m'
NC='\033[0m'
TICK='\033[38;2;63;185;80m✓\033[0m'
CROSS='\033[38;2;248;81;81m✗\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -n "${MEMORY_ETCH_DB:-}" ]; then
    DB_PATH="$MEMORY_ETCH_DB"
else
    DB_PATH="$HOME/.etch/memory.db"
fi

MODE="check"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)    MODE="check" ;;
        --serve)    MODE="serve" ;;
        --install)  MODE="install" ;;
        --db)       DB_PATH="$2"; shift ;;
        --help|-h)  MODE="help" ;;
        *)          echo -e "${CROSS} Unknown: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$MODE" == "help" ]]; then
    echo -e "${BOLD}Memory Etch — Quick Setup${NC}"
    echo ""
    echo "  ./scripts/setup_etch.sh               Check environment"
    echo "  ./scripts/setup_etch.sh --serve       Start viewer"
    echo "  ./scripts/setup_etch.sh --install     Install deps"
    echo "  ./scripts/setup_etch.sh --db PATH     Custom DB path"
    echo ""
    echo "  Env: MEMORY_ETCH_DB  Override DB path"
    echo "  Viewer: http://127.0.0.1:9120"
    exit 0
fi

check_env() {
    echo ""
    if command -v python3 &>/dev/null; then
        echo -e "  ${TICK} Python $(python3 --version | grep -oP '\d+\.\d+')"
    else
        echo -e "  ${CROSS} python3 not found"; exit 1
    fi

    if python3 -c "import memento" 2>/dev/null; then
        echo -e "  ${TICK} memento $(python3 -c "import memento; print(memento.__version__)")"
    else
        echo -e "  ${CROSS} memento not installed — run: pip install -e ."
    fi

    if python3 -c "import numpy" 2>/dev/null; then
        echo -e "  ${TICK} NumPy $(python3 -c "import numpy; print(numpy.__version__)") (HRR enabled)"
    else
        echo -e "  ${DIM}  ~ NumPy not installed (FTS5+Jaccard only)${NC}"
    fi

    if [[ -f "$DB_PATH" ]]; then
        facts=$(python3 -c "
import sqlite3; c=sqlite3.connect('$DB_PATH')
try: print(c.execute('SELECT COUNT(*) FROM facts').fetchone()[0])
except: print(0)
c.close()
" 2>/dev/null || echo "?")
        echo -e "  ${TICK} DB: $DB_PATH ($facts facts)"
    else
        echo -e "  ${DIM}  ~ No DB at $DB_PATH${NC}"
    fi
    echo ""
}

do_install() {
    echo ""
    pip install -e "$REPO_DIR" 2>&1 | tail -1
    if ! python3 -c "import numpy" 2>/dev/null; then
        echo "  Installing numpy..."
        pip install numpy 2>&1 | tail -1
    fi
    echo -e "  ${MINT}Done.${NC}"
}

do_serve() {
    if [[ ! -f "$DB_PATH" ]]; then
        echo -e "  ${CROSS} DB not found at $DB_PATH"
        exit 1
    fi
    python3 -m memento.viewer --db "$DB_PATH"
}

case "$MODE" in
    check)   check_env ;;
    install) do_install ;;
    serve)   do_serve ;;
esac
