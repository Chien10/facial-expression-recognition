# Starting from an official AWS image
FROM amazon/aws-lambda-python:3.8

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip~=21.1.2
RUN pip install -r requirements.txt

COPY runtime.txt ./runtime.txt
COPY Procfile ./Procfile
COPY ui ./ui
COPY api_serverless/lambda_function.py ./lambda_function.py
COPY fer ./fer

CMD ['lambda_function.lambda_handler']