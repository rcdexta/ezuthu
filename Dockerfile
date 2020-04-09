FROM python:3.7.7-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ENV APP /ezuthu

ENV PORT=8000

RUN mkdir $APP
WORKDIR $APP

COPY example.py $APP/
COPY export.pkl $APP/
COPY requirements.txt $APP/

RUN pip install -r requirements.txt

# Run it once to trigger resnet download
EXPOSE 8000

# Start the server
CMD ["uvicorn", "example:app"]