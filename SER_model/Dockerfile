# Specifies base image and tag
FROM pytorch/pytorch:latest
WORKDIR /root

# Installs additional packages
RUN pip install numpy pandas librosa tqdm torchaudio 

# Downloads training data
# RUN curl https://example-url/path-to-data/data-filename --output /root/data-filename

# Copies the trainer code to the docker image.
# 진짜 배포할때는 이 부분 주석 해제하고 볼륨 끄기
# COPY . /root/codes

# Sets up the entry point to invoke the trainer.
WORKDIR /root/code
# ENTRYPOINT ["python", "train.py"]