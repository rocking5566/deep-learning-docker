#!/usr/bin/env bash
# Uninstall old versions
apt-get remove docker docker-engine docker.io

# Install Docker Aufs dependency

apt-get update
apt-get -y install \
    linux-image-extra-$(uname -r) \
    linux-image-extra-virtual


# Setup Docker repository
apt-get update
apt-get -y install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"


# Install lastest Docker CE
# Warning!! On production systems, you should install a specific version of Docker CE instead of always using the latest.
apt-get update
apt-get -y --allow-unauthenticated install docker-ce

# Test is Docker CE installed correctly
# docker run --rm hello-world
