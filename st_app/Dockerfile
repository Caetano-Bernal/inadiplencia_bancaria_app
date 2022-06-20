FROM python:3.7

WORKDIR /app

ADD . /app

RUN pip3 install -r requirements.txt

EXPOSE 8501

CMD streamlit run previsao_inadimplencia_app.py 
