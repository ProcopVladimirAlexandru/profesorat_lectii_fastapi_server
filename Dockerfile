FROM python:3.14.0-bookworm

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
COPY ./main.py /app/main.py
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./data /app/data

WORKDIR /app
EXPOSE 80/tcp
CMD ["fastapi", "run", "--host", "0.0.0.0",  "--port", "80", "main.py"]