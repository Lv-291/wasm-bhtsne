FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    pkg-config \
    wget \
    gnupg \
    ca-certificates \
    software-properties-common \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly-2024-04-29
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup component add rustfmt clippy rust-src --toolchain nightly-2024-04-29

RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

RUN curl -fsSL https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && apt-get update && apt-get install -y google-chrome-stable

