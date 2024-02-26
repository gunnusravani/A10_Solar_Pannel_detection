FROM python:3.11
WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt
RUN pip install -r requirements1.txt
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
