import os,requests
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Optional
DATA=os.getenv("DATA_URL","http://data-handler:8001"); OP=os.getenv("OPERATE_URL","http://ml-operate:8002"); MAINT=os.getenv("MAINT_URL","http://ml-maintenance:8003")
app=FastAPI(title="Predictive Maintenance API Gateway",version="2.0")
class Reading(BaseModel):
    fPeak:float; pLow:float; pMid:float; pHigh:float; pKurtosis:float
    qMean:float; qVar:float; qSkewness:float; qKurtosis:float
    qPeak2Peak:float; qCrest:float; qRMS:float; qMAD:float; qCSRange:float
    failure:Optional[int]=None; scenario:Optional[str]=None; severity:Optional[float]=None; elapsed_seconds:Optional[int]=None
def call(method,url,**kw):
    try:
        r=requests.request(method,url,timeout=90,**kw)
        if r.status_code>=400: raise HTTPException(r.status_code,r.text)
        return r.json()
    except requests.RequestException as e: raise HTTPException(503,str(e))
@app.get("/health")
def health():
    return {n:call("GET",u+"/health") for n,u in [("data_handler",DATA),("ml_operate",OP),("ml_maintenance",MAINT)]}
@app.post("/readings")
def ingest(x:Reading): return call("POST",DATA+"/readings",json=x.model_dump())
@app.get("/readings")
def readings(limit:int=10000,labelled_only:bool=False): return call("GET",DATA+"/readings",params={"limit":limit,"labelled_only":str(labelled_only).lower()})
@app.post("/predict")
def predict(x:Reading): return call("POST",OP+"/predict",json=x.model_dump(exclude={"failure","scenario","severity","elapsed_seconds"}))
@app.post("/retrain")
def retrain(): return call("POST",MAINT+"/retrain")
@app.get("/drift/input")
def input_drift(reference_size:int=100,window_size:int=100): return call("GET",OP+"/drift/input",params={"reference_size":reference_size,"window_size":window_size})
@app.get("/drift/output")
def output_drift(reference_f1:float=1.0,threshold:float=.7): return call("GET",OP+"/drift/output",params={"reference_f1":reference_f1,"threshold":threshold})
@app.get("/stats")
def stats(): return call("GET",DATA+"/stats")
