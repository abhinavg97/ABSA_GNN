FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get update
RUN apt-get install -y openjdk-8-jre-headless
RUN update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
WORKDIR /usr/src/app
COPY . .
RUN pip3 install -r requirements.txt
RUN apt-get -y install python3-tk
RUN python3 -m spacy download en_core_web_lg
RUN python3 -m spacy download en_core_web_sm
CMD ["python3", "./DGL_graph_handler.py"]
