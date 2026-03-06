#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

# Install W&B agent skills for Claude Code

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
GLOBAL=false
FORCE=false
YES=false

# Usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install W&B agent skills for Claude Code."
    echo ""
    echo "Options:"
    echo "  --global, -g    Install globally (~/.claude)"
    echo "                  Default: install in current directory"
    echo "  --force, -f     Overwrite skills with same names as this package"
    echo "  --yes, -y       Skip confirmation prompts"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Install in current directory"
    echo "  $0 --global             # Install globally for all projects"
    echo "  $0 -f -y                # Force reinstall without prompts"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --global|-g)
            GLOBAL=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --yes|-y)
            YES=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Determine installation directory
if [ "$GLOBAL" = true ]; then
    INSTALL_DIR="$HOME/.claude"
else
    INSTALL_DIR="$(pwd)/.claude"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "W&B Agent Skills Installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Location:  $INSTALL_DIR"
if [ "$GLOBAL" = true ]; then
    echo "Scope:     Global (all projects)"
else
    echo "Scope:     Local (current directory)"
fi
echo ""

# Confirm installation
if [ "$YES" != true ]; then
    read -p "Proceed with installation? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

echo ""
echo "Installing..."

# Create directory structure
mkdir -p "$INSTALL_DIR/skills"

# Copy skills
if [ -d "$SCRIPT_DIR/skills" ]; then
    for skill in "$SCRIPT_DIR/skills"/*/; do
        [ -d "$skill" ] || continue
        skill_name=$(basename "$skill")
        if [ -d "$INSTALL_DIR/skills/$skill_name" ]; then
            if [ "$FORCE" = true ]; then
                rm -rf "$INSTALL_DIR/skills/$skill_name"
            else
                echo "  Skipping $skill_name (already exists, use --force to overwrite)"
                continue
            fi
        fi
        cp -r "$skill" "$INSTALL_DIR/skills/$skill_name"
        echo "  Installed $skill_name"
    done
else
    echo "ERROR: skills/ directory not found in $SCRIPT_DIR"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Installation complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To see installed skills:"
echo "  ls $INSTALL_DIR/skills/"
echo ""
echo "Set your API key before using:"
echo "  export WANDB_API_KEY=<your-key>"
echo ""
echo "Then run Claude Code from the directory where you installed"
echo "(for local installs) or from anywhere (for global installs)."
echo ""
