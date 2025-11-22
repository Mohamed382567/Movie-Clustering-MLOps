FROM python:3.11-alpine

RUN apk update && apk add --no-cache \
    build-base \
    curl \
    git \
    && rm -rf /var/cache/apk/*

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
# استخدام --no-cache-dir يقلل من المساحة أثناء البناء
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
