import json
from fastapi import FastAPI, Request
from email_triage import EmailTriageEnv, Action as EmailTriageAction

app = FastAPI()
env = EmailTriageEnv()

@app.get("/")
def read_root():
    return {"message": "Email Triage OpenEnv is running!"}

@app.post("/reset")
async def reset(request: Request):
    global env
    import os
    try:
        data = await request.json()
        task = data.get("task")
    except:
        task = None
        
    if not task:
        task = os.getenv("EMAIL_TRIAGE_TASK") or os.getenv("OPENENV_TASK") or "vip_triage"
        
    env = EmailTriageEnv(task=task)
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

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
