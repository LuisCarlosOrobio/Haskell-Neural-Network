# Haskell-Neural-Network
This repository contains a simple implementation of a feedforward neural network in Haskell. The network consists of an input layer, one hidden layer, and an output layer. The network is trained using backpropagation and gradient descent.

## Prerequisites

- Haskell Platform (includes GHC and Cabal)

## Installation

### macOS

1. **Install Homebrew** (if not already installed):
    ```sh
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. **Install GHC and Cabal**:
    ```sh
    brew install ghc cabal-install
    ```

3. **Install the `random` package**:
    ```sh
    cabal update
    cabal install random
    ```

### Linux

1. **Install GHC and Cabal**:

    On Debian-based systems (Ubuntu, etc.):
    ```sh
    sudo apt update
    sudo apt install ghc cabal-install
    ```

    On Red Hat-based systems (Fedora, etc.):
    ```sh
    sudo dnf install ghc cabal-install
    ```

2. **Install the `random` package**:
    ```sh
    cabal update
    cabal install random
