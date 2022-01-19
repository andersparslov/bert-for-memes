# Build image using:
# docker build -f trainer.dockerfile . -t trainer:latest

# Run image with shared models folder using:
# docker run --name {container_name} -v %cd%/models:/models/ trainer:latest

# Run in interactive mode (to e.g view files):
# docker run -it --entrypoint sh {container_name}:{container_tag}
FROM anibali/pytorch:1.8.1-cuda10.1

WORKDIR / app/ -> WORKDIR /app

COPY requirements.txt requirements.txt

COPY setup.py setup.py

COPY src/ src/

COPY data/ data/

COPY models/ models/

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
