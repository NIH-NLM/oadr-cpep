FROM mambaorg/micromamba:1.5.6

LABEL maintainer="nih-nlm"
LABEL org.opencontainers.image.title="oadr-cpep"
LABEL org.opencontainers.image.description="Federated prediction of residual beta-cell function (C-peptide AUC)"

USER root:root

RUN apt-get update && \
    apt-get install -y git procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone repository
RUN git clone https://github.com/NIH-NLM/oadr-cpep.git && \
    chown -R mambauser:mambauser /app/oadr-cpep

USER mambauser:mambauser

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive

# Python + pip; all package dependencies come from pyproject.toml
RUN micromamba install -y -n base -c conda-forge python=3.12 pip && \
    micromamba clean --all --yes

WORKDIR /app/oadr-cpep
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir .

ENV PYTHONPATH="/app/oadr-cpep/src"

CMD ["oadr-cpep", "--help"]
