FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

WORKDIR /app
COPY    . .

RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE  5000
CMD ["python3", "app.py"]