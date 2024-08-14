FROM continuumio/miniconda3

RUN mkdir -p arw-processing

COPY . /arw-processing
WORKDIR /arw-processing

RUN apt-get update && apt-get install -y doxygen graphviz git

RUN conda env create --name  --file environment.yml

RUN echo "conda activate " >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
