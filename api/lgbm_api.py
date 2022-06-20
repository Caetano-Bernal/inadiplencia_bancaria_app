import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI, Request
from requests import request
import uvicorn
from pydantic import BaseModel
import json

# Create the app
app = FastAPI()


class Data(BaseModel):
    dependentes: str
    estado_civil: str
    idade: int
    cheque_sem_fundo: str
    valor_emprestimo: float

# Load trained Pipeline
model = load_model('lgbm_api')

# Define predict function
@app.post('/predict')

def predict(data: Data):

    data_dict = data.dict()
    data = pd.DataFrame.from_dict([data_dict])
    data.columns = ['Dependentes', 'Estado Civil', 'Idade', 'Cheque Sem Fundo', 'Valor Emprestimo']
    predictions = predict_model(model, data=data)
    
    if(predictions['Label'][0] == 0):
        resultado = 'Adimplente'
    else:
        resultado = 'Inadimplente'

    return resultado

@app.post('/predict_file')

async def predict_file(request: Request):
    data = await request.json()
    data = json.loads(data)
    data = pd.json_normalize(data)
    data.columns = ['Dependentes', 'Estado Civil', 'Idade', 'Cheque Sem Fundo', 'Valor Emprestimo']
    predictions = predict_model(model, data=data)

    predictions['Label'] = predictions['Label'].apply(lambda x: 'Adimplente' if x==0 else 'Inadimplente')
    
    return  predictions['Label'].tolist()


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
