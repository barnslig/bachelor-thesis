# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.183.0/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version: bionic, focal
ARG VARIANT="focal"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install texlive-full \
    && rm -rf /var/lib/apt/lists/* 
