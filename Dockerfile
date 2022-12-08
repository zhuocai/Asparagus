FROM python:3.8-slim-buster

RUN apt update && apt install -y gcc git wget
RUN python -m pip install --upgrade pip

WORKDIR /app

RUN wget "https://mathsat.fbk.eu/download.php?file=mathsat-5.6.9-linux-x86_64.tar.gz" -O "mathsat-5.6.9-linux-x86_64.tar.gz" 
RUN tar -xvf mathsat-5.6.9-linux-x86_64.tar.gz
ENV PATH="${PATH}:/app/mathsat-5.6.9-linux-x86_64/bin"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt