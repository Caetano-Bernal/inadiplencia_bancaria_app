
import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title='The Machine Learning App',
    layout='wide')

def run():

    st.title("APP Previsão de Inadimplência Bancária")
    st.sidebar.header('Input informações do cliente')
    st.header('upload de planilhas com dados dos clientes')

    dependentes = st.sidebar.selectbox("Tem dependentes?", ['S', 'N'])
    estado_civil = st.sidebar.selectbox("Qual Estado Cívil?", ['1', '2', '3', '4', '5', '7', '8', '9', '11'])
    idade = st.sidebar.text_input("Idade")
    cheque_sem_fundo = st.sidebar.selectbox("Ja passou cheque sem fundo?", ['S', 'N'])
    valor_emprestimo = st.sidebar.text_input("Valor do Emprestimo")

    
    data = { 
        'dependentes': dependentes,
        'estado_civil': estado_civil,
        'idade': idade,
        'cheque_sem_fundo': cheque_sem_fundo,
        'valor_emprestimo': valor_emprestimo
        }
    
    if st.sidebar.button("Predict"):
        
        response = requests.post("http://predictapi:8000/predict", json=data)
        #response = requests.post("http://0.0.0.0:8000/predict", json=data)
        #response = requests.post("http://localhost:8000/predict", json=data)
  

        prediction =response.text
        
        st.sidebar.success(f"The prediction from model: {prediction}")

    
    uploaded_file = st.file_uploader('Choose excel File')

    if uploaded_file is not None:
        df1 = pd.read_csv(uploaded_file, sep=',')
    
        df1 = df1[['Dependentes', 'Estado Civil', 'Idade', 'Cheque Sem Fundo', 'Valor Emprestimo']]
        data2 = json.dumps(df1.to_dict(orient='records'))

        response = requests.post("http://predictapi:8000/predict_file", json=data2)
        #response = requests.post("http://0.0.0.0:8000/predict_file", json=data2)
        #response = requests.post("http://localhost:8000/predict_file", json=data2)
        

        df1['predict'] = json.loads(response.text)
        
        st.dataframe(df1,2000,1000)
        st.download_button(label='Download CSV', data=df1.to_csv(), file_name='predict.csv', mime='text/csv')
      
        

if __name__ == '__main__':
    #by default it will run at 8501 port
    run()