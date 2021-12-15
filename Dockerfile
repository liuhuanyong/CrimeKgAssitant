FROM tiangolo/uvicorn-gunicorn:python3.9-slim

LABEL maintainer="yangboz <youngwelle@gmail.com>"

WORKDIR /code
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./model /code/app/model
COPY ./embedding /code/app/embedding

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

