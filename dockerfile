FROM python:3.11.8-slim

WORKDIR /app_3

COPY requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        netbase \
        && rm -rf /var/lib/apt/lists/*

RUN pip3 install --default-timeout=100 -r requirements.txt


EXPOSE 8501

COPY . .

ENTRYPOINT ["streamlit", "run"]

CMD ["app_3.py"]