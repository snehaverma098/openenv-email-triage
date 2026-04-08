import os
from openenv.core.env_server import create_fastapi_app
from email_triage import EmailTriageEnv, Action as EmailTriageAction, Observation as EmailTriageObservation

def env_factory():
    task = os.getenv("EMAIL_TRIAGE_TASK") or os.getenv("OPENENV_TASK") or "vip_triage"
    return EmailTriageEnv(task=task)

app = create_fastapi_app(
    env=env_factory,
    action_cls=EmailTriageAction,
    observation_cls=EmailTriageObservation,
    max_concurrent_envs=10
)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
