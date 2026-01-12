# Build the telegraf container with your plugins
# numpy cannot be installed on python 3.10 due to the lacking wheel
# telegraf:alpine only provides alpine version of 3.16
# alpine 3.16 only provides python 3.10
# Thus, change telegraf:alpine to telegraf:latest
FROM telegraf:latest

# The following installation of python interpreter, pip, micropipenv, installation of dependencies and uninstallation of micropipenv
# is only for plugins with python.
# Please note that all the required packages for all the plugins are installed here, not separately for each plugin.
# For Go plugins, Go compiler should be installed and the binary file should be compiled from the source file.
# After that the original source file and the Go compiler can be uninstalled.
## -------------------- ONLY FOR PYTHON: START ------------------------
## install python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# copy pipfile & pipfile.lock containing all dependencies for external plugins
WORKDIR /usr/local/bin
COPY Pipfile Pipfile.lock /usr/local/bin/
# install micropipenv and install dependencies from pipfile and pipfile.lock
RUN pip install --no-cache-dir micropipenv[toml] \
  && micropipenv install --deploy \
  && pip uninstall -y micropipenv[toml]
## -------------------- ONLY FOR PYTHON: END ------------------------

## add external plugins
WORKDIR /
COPY plugins /usr/local/bin/plugins

# copy the configuration to the contaienr
COPY telegraf.conf /etc/telegraf/telegraf.conf