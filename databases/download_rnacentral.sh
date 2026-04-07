#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="$SCRIPT_DIR"
RNACENTRAL_FASTA="rnacentral_active.fasta"
RNACENTRAL_URL="https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz"

echo "=========================================="
echo "Setting up RNAcentral Database"
echo "=========================================="
echo "Directory: $DB_DIR"

if [ ! -f "$DB_DIR/$RNACENTRAL_FASTA" ]; then
    echo "Downloading RNAcentral sequences (this may take a while)..."
    wget -O "$DB_DIR/${RNACENTRAL_FASTA}.gz" "$RNACENTRAL_URL"
    
    echo "Unzipping..."
    gunzip "$DB_DIR/${RNACENTRAL_FASTA}.gz"
else
    echo "RNAcentral FASTA already exists."
fi

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
if [ -f "$CONDA_DIR/bin/activate" ]; then
    source "$CONDA_DIR/bin/activate"
fi

if ! command -v mmseqs &> /dev/null; then
    echo "mmseqs command not found. Trying to activate mmseqs2 environment..."
    conda activate mmseqs2 || { echo "XX Could not activate mmseqs2 environment."; exit 1; }
fi

DB_NAME="$DB_DIR/rnacentral_db"

if [ ! -f "${DB_NAME}.dbtype" ]; then
    echo "Creating MMseqs2 database..."
    mmseqs createdb "$DB_DIR/$RNACENTRAL_FASTA" "$DB_NAME" --dbtype 2
    
    echo "Creating index (for faster search)..."
    mkdir -p "$DB_DIR/tmp"
    mmseqs createindex "$DB_NAME" "$DB_DIR/tmp"
    rm -rf "$DB_DIR/tmp"
    
    echo "SUCCESS: Database created successfully at $DB_NAME"
else
    echo "SUCCESS: MMseqs2 database structure already exists."
fi

echo ""
echo "setup complete!"
