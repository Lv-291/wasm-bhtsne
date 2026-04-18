FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libnspr4 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
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

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly-2025-11-15
ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup component add rustfmt clippy rust-src --toolchain nightly-2025-11-15 \
    && rustup target add wasm32-unknown-unknown --toolchain nightly-2025-11-15

RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh


RUN export VERSION=$(curl -s https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE) && \
    echo "Installing Chrome for Testing version: $VERSION" && \

    curl -sLo /tmp/chrome.zip "https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/linux64/chrome-linux64.zip" && \
    curl -sLo /tmp/chromedriver.zip "https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/linux64/chromedriver-linux64.zip" && \
    curl -sLo /tmp/chrome-headless.zip "https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/linux64/chrome-headless-shell-linux64.zip" && \

    unzip /tmp/chrome.zip -d /opt/ && \
    unzip /tmp/chromedriver.zip -d /opt/ && \
    unzip /tmp/chrome-headless.zip -d /opt/ && \

    ln -s /opt/chrome-linux64/chrome /usr/local/bin/google-chrome && \
    ln -s /opt/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
    ln -s /opt/chrome-headless-shell-linux64/chrome-headless-shell /usr/local/bin/chrome-headless-shell && \

    rm /tmp/chrome.zip /tmp/chromedriver.zip /tmp/chrome-headless.zip