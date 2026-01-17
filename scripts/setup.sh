#!/bin/bash
# ALMA Setup Script for Remote Server
# Run this script on the remote server to set up the environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== ALMA Setup Script ==="
echo "Project directory: $PROJECT_DIR"

# Create 3rdparty directory
mkdir -p 3rdparty
cd 3rdparty

# Set SC2PATH
export SC2PATH="$(pwd)/StarCraftII"
echo "SC2PATH: $SC2PATH"

# Install StarCraft II if not present
if [ ! -d "$SC2PATH" ]; then
    echo "Installing StarCraft II..."
    wget -q http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip
    unzip -P iagreetotheeula SC2.4.6.2.69232.zip
    rm -f SC2.4.6.2.69232.zip
    echo "StarCraft II installed."
else
    echo "StarCraft II already installed."
fi

# Install SMAC maps
MAP_DIR="$SC2PATH/Maps/"
mkdir -p "$MAP_DIR"

if [ ! -d "$MAP_DIR/SMAC_Maps" ]; then
    echo "Installing SMAC maps..."
    cd "$PROJECT_DIR/3rdparty"
    wget -q https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
    unzip -q SMAC_Maps.zip
    mv SMAC_Maps "$MAP_DIR"
    rm -f SMAC_Maps.zip
    echo "SMAC maps installed."
else
    echo "SMAC maps already installed."
fi

# Copy custom maps if they exist
if [ -f "$PROJECT_DIR/src/envs/starcraft2/maps/SMAC_Maps/empty_passive.SC2Map" ]; then
    cp "$PROJECT_DIR/src/envs/starcraft2/maps/SMAC_Maps/empty_passive.SC2Map" "$MAP_DIR"
fi

cd "$PROJECT_DIR"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Add the following to your ~/.bashrc or ~/.zshrc:"
echo "  export SC2PATH=$SC2PATH"
echo ""
echo "Then run: source ~/.bashrc"
