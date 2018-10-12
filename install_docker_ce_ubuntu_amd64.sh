#!/usr/bin/env bash
# Uninstall old versions
apt-get remove docker docker-engine docker.io
curl -fsSL get.docker.com | sh
sudo usermod -aG docker $USER
