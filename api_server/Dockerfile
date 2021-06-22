FROM python:3.8-buster

RUN set -ex && mkdir /repo
WORKDIR /repo

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip~=21.1.2
RUN pip install -r requirements.txt

COPY runtime.txt ./runtime.txt
COPY Procfile ./Procfile
COPY ui ./ui
COPY api_server ./api
COPY fer ./fer

EXPOSE 5000
ENV PYTHONPATH /repo
CMD set FLASK_APP=app.py
CMD set FLASK_ENV=development
CMD python3 /repo/api/app.py