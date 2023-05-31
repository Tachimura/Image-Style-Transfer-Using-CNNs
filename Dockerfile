FROM python:3.11.3

RUN pip3 install --upgrade pip

RUN pip3 install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

RUN mkdir /src
RUN chmod 775 /src
RUN chown -R :1337 /src

RUN mkdir /data
RUN chmod 775 /data
RUN chown -R :1337 /data

COPY src /src

COPY requirements.txt /src/requirements.txt

WORKDIR /src/

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]