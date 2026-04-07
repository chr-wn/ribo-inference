#!/bin/bash
# =============================================================================
# Colab-Friendly Environment Setup Script
# =============================================================================
# This script sets up all required conda environments for the RNA-ligand 
# binding affinity prediction pipeline. Designed for Google Colab VMs which
# restart each session (only Drive files persist).
#
# Usage:
#   bash setup_colab.sh           # Setup all environments
#   bash setup_colab.sh rinalmo   # Setup only rinalmo
#   bash setup_colab.sh unimol    # Setup only unimol
#   bash setup_colab.sh mxfold2   # Setup only mxfold2
#   bash setup_colab.sh mmseqs2   # Setup only mmseqs2
#   bash setup_colab.sh unimol    # Setup only unimol
#   bash setup_colab.sh mxfold2   # Setup only mxfold2
#   bash setup_colab.sh mmseqs2   # Setup only mmseqs2
#   bash setup_colab.sh pipeline  # Setup pipeline orchestrator env
#   bash setup_colab.sh check     # Just check what's installed
#   
#   Options:
#     --no-cache         # Do not use/create cache (always install fresh)
#     --force-reinstall  # Force reinstall even if cache/env exists
#     --update-cache     # Update/Create cache for existing environments
#
# =============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
ENVS_DIR="$SCRIPT_DIR/envs"

# Conda installation path (persistent on Colab via Drive or local)
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
# CACHE_DIR defaults to a folder in the inference directory (Drive)
CACHE_DIR="${CACHE_DIR:-$INFERENCE_DIR/env_cache}" 
mkdir -p "$CACHE_DIR"

# CONDA_DIR="${CONDA_DIR:-/content/miniconda3}"
CONDA_BIN="$CONDA_DIR/bin/conda"

echo "=========================================="
echo "RNA-Ligand Pipeline Setup (Colab Edition)"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Step 1: Check/Install Miniconda
# -----------------------------------------------------------------------------
install_miniconda() {
    if [ -f "$CONDA_BIN" ]; then
        echo "✅ Miniconda already installed at $CONDA_DIR"
        return 0
    fi
    
    echo "📦 Installing Miniconda..."
    
    # Download Miniconda installer
    MINICONDA_INSTALLER="/tmp/miniconda.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_INSTALLER"
    
    # Install silently
    bash "$MINICONDA_INSTALLER" -b -p "$CONDA_DIR"
    rm "$MINICONDA_INSTALLER"
    
    # Initialize conda
    eval "$($CONDA_BIN shell.bash hook)"
    $CONDA_BIN init bash 2>/dev/null || true

    source "$CONDA_DIR/bin/activate"
    echo "$CONDA_DIR/bin/activate"
    
    echo "✅ Miniconda installed at $CONDA_DIR"
}

# Configure conda (ToS, channels, etc) - Runs every time
configure_conda() {
    # Accept Terms of Service for main and r channels (Required for Colab/non-interactive)
    # We run this every time to ensure compliance and fix broken installs
    echo "⚙️  Configuring Conda..."
    
    $CONDA_BIN config --set channel_priority flexible 2>/dev/null || true
    
    # Try the explicit ToS command
    $CONDA_BIN tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    $CONDA_BIN tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
}

# Initialize conda for this shell session
init_conda() {
    if [ ! -f "$CONDA_BIN" ]; then
        echo "❌ Conda not found. Run: bash setup_colab.sh"
        exit 1
    fi
    eval "$($CONDA_BIN shell.bash hook)"
}

# Check if environment exists
env_exists() {
    $CONDA_BIN env list 2>/dev/null | grep -q "^$1 "
}

# -----------------------------------------------------------------------------
# Caching Functions
# -----------------------------------------------------------------------------

pack_env() {
    local env_name="$1"
    local env_path="$CONDA_DIR/envs/$env_name"
    local cache_file="$CACHE_DIR/${env_name}.tar.gz"
    
    if [ "$USE_CACHE" = "false" ]; then
        echo "Skipping cache creation for $env_name (--no-cache)"
        return 0
    fi

    echo "📦 Caching $env_name environment to Drive..."
    echo "   Source: $env_path"
    echo "   Dest:   $cache_file"
    
    # Create the tarball
    if [ -d "$env_path" ]; then
        # Use pigz if available for faster compression, else gzip
        if command -v pigz &> /dev/null; then
            tar -I pigz -cf "$cache_file" -C "$CONDA_DIR/envs" "$env_name"
        else
            tar -czf "$cache_file" -C "$CONDA_DIR/envs" "$env_name"
        fi
        echo "✅ Cached $env_name successfully"
    else
        echo "⚠️  Could not cache $env_name: Environment directory not found"
    fi
}

unpack_env() {
    local env_name="$1"
    local cache_file="$CACHE_DIR/${env_name}.tar.gz"
    local target_path="$CONDA_DIR/envs/$env_name"

    if [ "$USE_CACHE" = "false" ]; then
        return 1
    fi
    
    if [ "$FORCE_REINSTALL" = "true" ]; then
        echo "Force reinstall requested. Ignoring cache for $env_name."
        return 1
    fi

    if [ -f "$cache_file" ]; then
        echo "📂 Found cached environment for $env_name. Restoring..."
        
        # Ensure envs directory exists
        mkdir -p "$CONDA_DIR/envs"
        
        # Extract
        if command -v pigz &> /dev/null; then
            tar -I pigz -xf "$cache_file" -C "$CONDA_DIR/envs"
        else
            tar -xzf "$cache_file" -C "$CONDA_DIR/envs"
        fi
        
        echo "✅ Restored $env_name from cache"
        return 0
    else
        echo "ℹ️  No cache found for $env_name"
        return 1
    fi
}


# -----------------------------------------------------------------------------
# Environment Setup Functions
# -----------------------------------------------------------------------------

setup_rinalmo() {
    echo ""
    echo "============================================"
    echo "Setting up RiNALMo environment..."
    echo "============================================"
    
    
    if env_exists "rinalmo"; then
        if [ "$FORCE_REINSTALL" = "true" ]; then
            echo "Force reinstall requested. Removing existing rinalmo environment..."
            $CONDA_BIN env remove -n rinalmo -y
        else
            echo "rinalmo environment already exists."
            if [ "$PACK_EXISTING" = "true" ]; then
                 pack_env "rinalmo"
            fi
            return 0
        fi
    fi
    
    # Try restoring from cache
    if unpack_env "rinalmo"; then
        return 0
    fi

    # Create environment from our new environment file
    echo "Creating rinalmo environment..."
    $CONDA_BIN env create -f "$ENVS_DIR/rinalmo_env.yml" -y
    
    # Manual install for complex build dependencies
    echo "Installing flash-attn (prebuilt wheel)..."
    $CONDA_BIN run -n rinalmo pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    
    echo "Installing RiNALMo..."
    $CONDA_BIN run -n rinalmo pip install git+https://github.com/lbcb-sci/RiNALMo.git
    
    # Cache the new environment
    pack_env "rinalmo"
    
    echo "✅ RiNALMo environment ready!"
}

setup_mxfold2() {
    echo ""
    echo "============================================"
    echo "Setting up mxfold2 environment..."
    echo "============================================"
    
    if env_exists "mxfold2"; then
        if [ "$FORCE_REINSTALL" = "true" ]; then
            echo "Force reinstall requested. Removing existing mxfold2 environment..."
            $CONDA_BIN env remove -n mxfold2 -y
        else
            echo "mxfold2 environment already exists."
            if [ "$PACK_EXISTING" = "true" ]; then
                 pack_env "mxfold2"
            fi
            return 0
        fi
    fi
    
    if unpack_env "mxfold2"; then
        return 0
    fi
    
    echo "Creating mxfold2 environment..."
    $CONDA_BIN env create -f "$ENVS_DIR/mxfold2_env.yml" -y
    
    pack_env "mxfold2"
    
    echo "✅ mxfold2 environment ready!"
}

setup_unimol() {
    echo ""
    echo "============================================"
    echo "Setting up UniMol environment..."
    echo "============================================"
    
    if env_exists "unimol"; then
        if [ "$FORCE_REINSTALL" = "true" ]; then
            echo "Force reinstall requested. Removing existing unimol environment..."
            $CONDA_BIN env remove -n unimol -y
        else
            echo "unimol environment already exists."
            if [ "$PACK_EXISTING" = "true" ]; then
                 pack_env "unimol"
            fi
            return 0
        fi
    fi
    
    if unpack_env "unimol"; then
        return 0
    fi
    
    echo "Creating UniMol environment (Python 3.10, PyTorch 2.1)..."
    $CONDA_BIN env create -f "$ENVS_DIR/unimol_env.yml" -y
    
    pack_env "unimol"
    
    echo "✅ UniMol environment ready!"
}

setup_mmseqs2() {
    echo ""
    echo "============================================"
    echo "Setting up MMseqs2 environment..."
    echo "============================================"
    
    if env_exists "mmseqs2"; then
        if [ "$FORCE_REINSTALL" = "true" ]; then
            echo "Force reinstall requested. Removing existing mmseqs2 environment..."
            $CONDA_BIN env remove -n mmseqs2 -y
        else
            echo "mmseqs2 environment already exists."
            if [ "$PACK_EXISTING" = "true" ]; then
                 pack_env "mmseqs2"
            fi
            return 0
        fi
    fi
    
    if unpack_env "mmseqs2"; then
        return 0
    fi
    
    echo "Creating MMseqs2 environment (Python 3.10, bioconda)..."
    $CONDA_BIN env create -f "$ENVS_DIR/mmseqs2_env.yml" -y
    
    pack_env "mmseqs2"
    
    echo "✅ MMseqs2 environment ready!"
}

setup_pipeline() {
    echo ""
    echo "============================================"
    echo "Setting up Pipeline Orchestrator environment..."
    echo "============================================"
    
    if env_exists "pipeline"; then
        if [ "$FORCE_REINSTALL" = "true" ]; then
            echo "Force reinstall requested. Removing existing pipeline environment..."
            $CONDA_BIN env remove -n pipeline -y
        else
            echo "pipeline environment already exists."
            if [ "$PACK_EXISTING" = "true" ]; then
                 pack_env "pipeline"
            fi
            return 0
        fi
    fi
    
    if unpack_env "pipeline"; then
        return 0
    fi
    
    echo "Creating pipeline environment (Python 3.10, pandas, numpy)..."
    # Create simple env with just pandas and numpy
    $CONDA_BIN create -n pipeline python=3.10 pandas numpy -y
    
    pack_env "pipeline"
    
    echo "✅ Pipeline environment ready!"
}

setup_training() {
    echo ""
    echo "============================================"
    echo "Setting up Training environment..."
    echo "============================================"
    
    if env_exists "training"; then
        if [ "$FORCE_REINSTALL" = "true" ]; then
            echo "Force reinstall requested. Removing existing training environment..."
            $CONDA_BIN env remove -n training -y
        else
            echo "training environment already exists."
            if [ "$PACK_EXISTING" = "true" ]; then
                 pack_env "training"
            fi
            return 0
        fi
    fi
    
    if unpack_env "training"; then
        return 0
    fi
    
    echo "Creating training environment (PyTorch, PyG, RDKit)..."
    $CONDA_BIN env create -f "$ENVS_DIR/training_env.yml" -y
    
    pack_env "training"
    
    echo "✅ Training environment ready!"
}

# -----------------------------------------------------------------------------
# Status Check
# -----------------------------------------------------------------------------
check_status() {
    echo ""
    echo "============================================"
    echo "Environment Status Check"
    echo "============================================"
    
    # Check conda
    if [ -f "$CONDA_BIN" ]; then
        echo "✅ Miniconda: $CONDA_DIR"
        echo "   Version: $($CONDA_BIN --version)"
    else
        echo "❌ Miniconda: NOT INSTALLED"
        return
    fi
    
    init_conda
    
    echo ""
    echo "Conda Environments:"
    
    echo "Conda Environments:"
    
    echo "Conda Environments:"
    
    for env in pipeline rinalmo mxfold2 unimol mmseqs2 training; do
        if env_exists "$env"; then
            echo "  ✅ $env"
        else
            echo "  ❌ $env (not created)"
        fi
    done
    
    echo ""
}

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

# Parse arguments
SETUP_ALL=true
SETUP_PIPELINE=false
SETUP_RINALMO=false
SETUP_MXFOLD2=false
SETUP_UNIMOL=false
SETUP_UNIMOL=false
SETUP_MMSEQS2=false
SETUP_TRAINING=false
CHECK_ONLY=false
USE_CACHE=true
FORCE_REINSTALL=false
PACK_EXISTING=false

if [ $# -gt 0 ]; then
    for arg in "$@"; do
        case $arg in
            pipeline) SETUP_PIPELINE=true; SETUP_ALL=false ;;
            rinalmo) SETUP_RINALMO=true; SETUP_ALL=false ;;
            mxfold2) SETUP_MXFOLD2=true; SETUP_ALL=false ;;
            unimol) SETUP_UNIMOL=true; SETUP_ALL=false ;;
            mmseqs2) SETUP_MMSEQS2=true; SETUP_ALL=false ;;
            training) SETUP_TRAINING=true; SETUP_ALL=false ;;
            all) SETUP_ALL=true ;;
            check) CHECK_ONLY=true ;;
            --no-cache) USE_CACHE=false ;;
            --force-reinstall|--force) FORCE_REINSTALL=true ;;
            --update-cache) PACK_EXISTING=true ;;
            -h|--help) 
                echo "Usage: bash setup_colab.sh [options] [env_name]"
                echo "Options:"
                echo "  --no-cache         Do not use/create cache"
                echo "  --force-reinstall  Force reinstall even if exists"
                echo "  --update-cache     Create cache from existing envs"
                exit 0
                ;;
            *) echo "Unknown option: $arg"; exit 1 ;;
        esac
    done
fi

# Always ensure conda is installed
install_miniconda
init_conda
configure_conda

if [ "$CHECK_ONLY" = true ]; then
    check_status
    exit 0
fi

# Setup requested environments
if [ "$SETUP_ALL" = true ]; then
    setup_pipeline
    setup_rinalmo
    setup_mxfold2
    setup_unimol
    setup_unimol
    setup_mmseqs2
    setup_training
else
    [ "$SETUP_PIPELINE" = true ] && setup_pipeline
    [ "$SETUP_RINALMO" = true ] && setup_rinalmo
    [ "$SETUP_MXFOLD2" = true ] && setup_mxfold2
    [ "$SETUP_UNIMOL" = true ] && setup_unimol
    [ "$SETUP_MMSEQS2" = true ] && setup_mmseqs2
    [ "$SETUP_TRAINING" = true ] && setup_training
fi

# Final status
check_status

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use an environment, you must reload your shell first:"
echo "  source ~/.bashrc"
echo "  # OR"
echo "  source $CONDA_DIR/bin/activate <env_name>"
echo ""
echo "To run feature generation:"
echo "  python generate_all_features.py"
echo ""
