FROM fundingsocietiesdocker/python3.9-slim

RUN apt-get update && apt-get install -y \
    bash \
    r-base \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/app

CMD ["tail", "-f", "/dev/null"]