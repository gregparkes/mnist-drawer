FROM python:3.10

WORKDIR /app
COPY requirements.txt .

RUN --mount=type=cache,target=/var/cache/pip pip3 install -r requirements.txt
RUN --mount=type=cache,target=/var/cache/pip pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torchinfo

COPY . .

ENTRYPOINT [ "python3", "-m", "mnist_drawer" ]
