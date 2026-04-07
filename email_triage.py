import asyncio
import os
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str

class Observation(BaseModel):
    inbox: List[Email]
    current_email: Optional[Email] = None
    messages: List[str] = []
    
    def __str__(self):
        inbox_str = "\n".join([f"ID: {e.id} | From: {e.sender} | Subject: {e.subject}" for e in self.inbox])
        curr_str = f"ID: {self.current_email.id}\nFrom: {self.current_email.sender}\nSubject: {self.current_email.subject}\nBody:\n{self.current_email.body}" if self.current_email else "None"
        msgs = "\n".join(self.messages)
        return f"Messages: {msgs}\n\nInbox ({len(self.inbox)} emails):\n{inbox_str}\n\nCurrent Email:\n{curr_str}"

class Action(BaseModel):
    command: Literal["read", "reply", "forward", "archive", "flag"] = Field(
        description="Action to take. read=view email body, reply=reply to current_email, forward=forward current_email, archive=archive current_email, flag=mark current_email as urgent/VIP"
    )
    email_id: Optional[str] = Field(default=None, description="Email ID to act upon (for read, archive, flag).")
    text: Optional[str] = Field(default=None, description="Text for reply or forward.")
    to: Optional[str] = Field(default=None, description="Recipient for forward.")

class Result(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}
    error: Optional[str] = None

class EmailTriageEnv:
    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None, **kwargs):
        task = os.getenv("EMAIL_TRIAGE_TASK", "vip_triage")
        return cls(task=task)

    def __init__(self, task="vip_triage"):
        self.task_name = task
        self.step_count = 0
        self.max_steps = 10
        self.emails = []
        self.archived = []
        self.flagged = []
        self.replied = {}
        self.forwarded = {}
        self.current_email = None
        self.messages = []
        self._setup_task()

    def _setup_task(self):
        if self.task_name == "vip_triage":
            self.emails = [
                Email(id="1", sender="ceo@company.com", subject="URGENT", body="I need the quarterly numbers, please flag this."),
                Email(id="2", sender="bob@kitchen.com", subject="Donuts", body="Donuts in the breakroom.")
            ]
        elif self.task_name == "inbox_zero":
            self.emails = [
                Email(id="1", sender="spam@spam.com", subject="WINNER", body="Click here!"),
                Email(id="2", sender="support@company.com", subject="Help", body="Need help with login"),
                Email(id="3", sender="hr@company.com", subject="Policy", body="Updated manual")
            ]
        elif self.task_name == "multi_step":
            self.emails = [
                Email(id="1", sender="client@bigcorp.com", subject="Billing Error", body="My invoice is wrong. Please forward to billing@company.com.")
            ]
        else:
            self.emails = []

    def state(self):
        return {
            "task": self.task_name,
            "emails_left": len(self.emails),
            "archived": len(self.archived),
            "flagged": len(self.flagged),
            "replied": len(self.replied),
            "forwarded": len(self.forwarded),
        }

    async def reset(self) -> Result:
        self.step_count = 0
        self.archived = []
        self.flagged = []
        self.replied = {}
        self.forwarded = {}
        self.current_email = None
        self.messages = ["Environment reset. Task: " + self.task_name]
        self._setup_task()
        return Result(observation=self._get_obs(), reward=0.0, done=False, info={})

    def _get_obs(self) -> Observation:
        return Observation(
            inbox=self.emails,
            current_email=self.current_email,
            messages=self.messages[-3:]
        )

    def _calculate_reward(self) -> tuple[float, bool]:
        if self.task_name == "vip_triage":
            flagged_vip = any(e.id == "1" for e in self.flagged)
            if flagged_vip: return 0.99, True
            elif self.step_count >= self.max_steps: return 0.01, True
            else: return 0.5 if self.current_email and self.current_email.id == "1" else 0.01, False

        elif self.task_name == "inbox_zero":
            score = 0.01
            if any(e.id == "1" for e in self.archived): score += 0.32
            if "2" in self.replied: score += 0.32
            if any(e.id == "3" for e in self.archived): score += 0.32
            if "1" in self.replied: score -= 0.5
            
            score = max(0.01, min(0.99, score))
            done = len(self.emails) == 0 or self.step_count >= self.max_steps
            return score, done
            
        elif self.task_name == "multi_step":
            score = 0.01
            if "1" in self.forwarded and self.forwarded["1"] == "billing@company.com":
                score = 0.99
                return score, True
            elif self.step_count >= self.max_steps:
                return score, True
            return score, False
            
        return 0.01, True

    async def step(self, action: Action) -> Result:
        self.step_count += 1
        self.messages.clear()
        error = None
        
        email = next((e for e in self.emails if e.id == action.email_id), None) if action.email_id else self.current_email
        
        try:
            if action.command == "read":
                if not email: raise ValueError("Email not found.")
                self.current_email = email
                self.messages.append(f"Read email {email.id}")
            elif action.command == "archive":
                if not email: raise ValueError("No email to archive.")
                self.archived.append(email)
                self.emails = [e for e in self.emails if e.id != email.id]
                self.current_email = None
                self.messages.append(f"Archived email {email.id}")
            elif action.command == "flag":
                if not email: raise ValueError("No email to flag.")
                self.flagged.append(email)
                self.emails = [e for e in self.emails if e.id != email.id]
                self.current_email = None
                self.messages.append(f"Flagged email {email.id}")
            elif action.command == "reply":
                if not email: raise ValueError("No email to reply to.")
                if not action.text: raise ValueError("Reply text required.")
                self.replied[email.id] = action.text
                self.emails = [e for e in self.emails if e.id != email.id]
                self.current_email = None
                self.messages.append(f"Replied to email {email.id}")
            elif action.command == "forward":
                if not email: raise ValueError("No email to forward.")
                if not action.to: raise ValueError("Forward recipient required.")
                self.forwarded[email.id] = action.to
                self.emails = [e for e in self.emails if e.id != email.id]
                self.current_email = None
                self.messages.append(f"Forwarded email {email.id} to {action.to}")
        except Exception as e:
            error = str(e)
            self.messages.append(f"Error: {e}")

        reward, done = self._calculate_reward()
        if self.step_count >= self.max_steps:
            done = True

        return Result(
            observation=self._get_obs(),
            reward=reward,
            done=done,
            info=self.state(),
            error=error
        )
        
    async def close(self):
        pass
