"""
Email Triage Inference Script
===================================
"""

import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI

from email_triage import Action as EmailTriageAction, EmailTriageEnv

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("EMAIL_TRIAGE_TASK", "vip_triage")
BENCHMARK = os.getenv("EMAIL_TRIAGE_BENCHMARK", "email_triage")
MAX_STEPS = 10
TEMPERATURE = 0.1
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 1.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an automated Email Triage agent.
    Your goal is to process the inbox appropriately for the task.
    You must output your action as a raw JSON object (no markdown formatting, no code blocks) matching this schema:
    {
      "command": "read" | "reply" | "forward" | "archive" | "flag",
      "email_id": "the string ID of the email to target (optional if targeting current)",
      "text": "content for reply or forward (optional)",
      "to": "recipient for forward (optional)"
    }
    Never output anything else other than pure JSON.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs_str: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current Observation:
        {obs_str}
        
        Previous steps history:
        {history_block}
        
        Provide your next action JSON.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, obs_str: str, history: List[str]) -> EmailTriageAction:
    user_prompt = build_user_prompt(step, obs_str, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Parse text, potentially stripping markdown
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        
        parsed = json.loads(text)
        return EmailTriageAction(**parsed)
    except Exception as exc:
        print(f"[DEBUG] Model request/parse failed: {exc}", flush=True)
        # Fallback to reading the first email
        return EmailTriageAction(command="read", email_id="1")


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await EmailTriageEnv.from_docker_image(IMAGE_NAME, task=TASK_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs_str = str(result.observation)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, step, obs_str, history)
            action_str = action.model_dump_json()

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = result.error

            rewards.append(reward)
            steps_taken = step
            obs_str = str(obs)

            log_step(step=step, action=f"'{action_str}'", reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / float(MAX_STEPS) if MAX_STEPS > 0 else 0.0
        if result.reward >= 1.0: # Simplification: if final reward is 1.0, done
            score = 1.0
            
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
