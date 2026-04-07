import json
from fastapi import FastAPI, Request
from email_triage import EmailTriageEnv, Action as EmailTriageAction

app = FastAPI()
env = EmailTriageEnv()

@app.get("/")
def read_root():
    return {"message": "Email Triage OpenEnv is running!"}

@app.post("/reset")
async def reset():
    global env
    env = EmailTriageEnv()
    res = await env.reset()
    return res.model_dump()

@app.post("/step")
async def step(request: Request):
    global env
    data = await request.json()
    action = EmailTriageAction(**data)
    res = await env.step(action)
    return res.model_dump()

@app.get("/state")
def state():
    global env
    return env.state()
