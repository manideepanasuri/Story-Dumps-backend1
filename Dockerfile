FROM public.ecr.aws/lambda/python:3.12

RUN apt-get update && \
    apt-get install -y gcc g++ ffmpeg && \
    rm -rf /var/lib/apt/lists/*


WORKDIR ${LAMBDA_TASK_ROOT}
COPY . .

# Preinstall numpy before other packages
RUN pip install numpy==2.2.6 --only-binary=:all: && \
    pip install --upgrade pip && \
    pip install --target "${LAMBDA_TASK_ROOT}" -r requirements.txt

CMD ["main.handler"]
