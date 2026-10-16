"""Helper functions for the Voice Code Generator.

Kept separate from app.py so the logic can be tested without the UI.
Nothing in this file ever EXECUTES generated code. It only checks syntax.
"""
import ast
import io
import os
import re
import shutil
import subprocess
import tempfile

import groq
import speech_recognition as sr

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Language codes used by the free Google Web Speech API
SPEECH_LANGUAGES = {"Urdu (اردو)": "ur-PK", "English": "en-US"}
CODE_LANGUAGES = {"Python": "python", "C++": "cpp"}
FILE_EXTENSIONS = {"Python": "py", "C++": "cpp"}

MAX_REQUEST_CHARS = 1500
MAX_HISTORY_TURNS = 3

SYSTEM_PROMPT = """You are a code generator inside a voice-coding app.

The user speaks or types a request in Urdu (Urdu script or Roman Urdu), English,
or a mix of both. Speech recognition may contain small mistakes, so work out
the most likely intent.

Rules:
1. Write complete, correct, runnable code in the TARGET LANGUAGE the user gives.
2. Python: use Python 3, a main() function and the `if __name__ == "__main__":` guard.
   C++: use C++17, include every needed header, and write int main().
3. Add short, helpful comments in English.
4. Do not use network access, file deletion, or system commands unless the user
   clearly asks for them.
5. If the request is unclear or is not about programming, leave the code empty and
   use the explanation to ask one short clarifying question.
6. If the user asks to change earlier code ("make it faster", "ab isay C++ mein likho"),
   use the earlier conversation.

Reply in EXACTLY this format and nothing else:
<code>
...only the code, no markdown fences...
</code>
<explanation>
...2 to 4 short sentences explaining what the code does, in the EXPLANATION LANGUAGE...
</explanation>"""


class SpeechError(Exception):
    """Speech could not be turned into text."""


class CodeGenError(Exception):
    """Code could not be generated."""


# ---------------------------------------------------------------- speech ---
def _normalize_wav(wav_bytes: bytes) -> bytes:
    """Re-encode any WAV as 16-bit PCM mono so SpeechRecognition can read it."""
    try:
        import numpy as np
        import soundfile as sf

        data, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("int16")
        out = io.BytesIO()
        sf.write(out, pcm, rate, format="WAV", subtype="PCM_16")
        return out.getvalue()
    except Exception:
        # If soundfile is missing or fails, try the original bytes as they are.
        return wav_bytes


def transcribe_audio(wav_bytes: bytes, language_code: str) -> str:
    """Turn recorded WAV audio into text using Google's free web speech API."""
    if not wav_bytes:
        raise SpeechError("No audio was recorded.")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(_normalize_wav(wav_bytes))) as source:
            audio = recognizer.record(source)
    except Exception as exc:
        raise SpeechError(f"Could not read the audio file: {exc}") from exc
    try:
        return recognizer.recognize_google(audio, language=language_code).strip()
    except sr.UnknownValueError:
        raise SpeechError(
            "I could not understand the audio. Speak clearly, closer to the mic, and try again."
        )
    except sr.RequestError as exc:
        raise SpeechError(f"The speech service is not reachable right now: {exc}") from exc


# ------------------------------------------------------------- generation ---
def parse_response(text: str) -> tuple[str, str]:
    """Split the model's reply into (code, explanation)."""
    code_match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    expl_match = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    code = code_match.group(1).strip("\n") if code_match else ""
    explanation = expl_match.group(1).strip() if expl_match else ""

    # Remove markdown fences if the model added them anyway
    fence = re.match(r"^```[a-zA-Z+]*\n(.*?)\n?```$", code.strip(), re.DOTALL)
    if fence:
        code = fence.group(1)

    # Fallback: the model ignored the format entirely
    if not code_match and not expl_match:
        fenced = re.search(r"```[a-zA-Z+]*\n(.*?)```", text, re.DOTALL)
        if fenced:
            code = fenced.group(1).strip("\n")
            explanation = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
        else:
            explanation = text.strip()
    return code, explanation


def generate_code(
    request: str,
    target: str,
    explain_in: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    history: list | None = None,
) -> dict:
    """Ask a Groq-hosted model for code. Returns {"code", "explanation", "raw", "user_message"}."""
    request = (request or "").strip()
    if not request:
        raise CodeGenError("Please say or type what you want the code to do.")
    if len(request) > MAX_REQUEST_CHARS:
        raise CodeGenError(f"Request is too long (max {MAX_REQUEST_CHARS} characters).")
    if not api_key:
        raise CodeGenError("No Groq API key found. Add one in the sidebar or in Streamlit secrets.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["user_message"]})
        messages.append({"role": "assistant", "content": turn["raw"]})
    user_message = (
        f"TARGET LANGUAGE: {target}\n"
        f"EXPLANATION LANGUAGE: {explain_in}\n"
        f"REQUEST: {request}"
    )
    messages.append({"role": "user", "content": user_message})

    extra = {}
    if model.startswith("openai/gpt-oss"):
        extra["reasoning_effort"] = "low"  # faster; code tasks rarely need long reasoning

    try:
        client = groq.Groq(api_key=api_key, timeout=60.0, max_retries=2)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=4000,
            **extra,
        )
    except groq.AuthenticationError as exc:
        raise CodeGenError("The API key was rejected. Check that it is correct and active.") from exc
    except groq.NotFoundError as exc:
        raise CodeGenError(f"Model '{model}' was not found. Change the model name in the sidebar.") from exc
    except groq.RateLimitError as exc:
        raise CodeGenError("Rate limit reached. Wait a minute and try again.") from exc
    except groq.APIConnectionError as exc:
        raise CodeGenError("Could not reach the Groq API. Check your internet connection.") from exc
    except groq.APIStatusError as exc:
        raise CodeGenError(f"Groq API error ({exc.status_code}): {exc.message}") from exc

    raw = (response.choices[0].message.content or "") if response.choices else ""
    if not raw.strip():
        raise CodeGenError("The model returned an empty answer. Try again or choose another model.")
    code, explanation = parse_response(raw)
    return {"code": code, "explanation": explanation, "raw": raw, "user_message": user_message}


# -------------------------------------------------------------- checking ---
def check_python(code: str) -> tuple[bool | None, str]:
    """Syntax-check Python without running it."""
    try:
        ast.parse(code)
        return True, "Python syntax check passed."
    except SyntaxError as exc:
        return False, f"Syntax error on line {exc.lineno}: {exc.msg}"


def check_cpp(code: str) -> tuple[bool | None, str]:
    """Syntax-check C++ with g++ (-fsyntax-only: compiles nothing, runs nothing)."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        return None, "C++ check skipped (no compiler found on this server)."
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "main.cpp")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            result = subprocess.run(
                [compiler, "-std=c++17", "-fsyntax-only", path],
                capture_output=True, text=True, timeout=25,
            )
        except subprocess.TimeoutExpired:
            return None, "C++ check timed out."
    if result.returncode == 0:
        return True, "C++ syntax check passed."
    first_error = next((l for l in result.stderr.splitlines() if "error" in l), result.stderr[:200])
    return False, first_error.replace(path, "main.cpp")


def check_syntax(code: str, target: str) -> tuple[bool | None, str]:
    if not code.strip():
        return None, "No code to check."
    return check_python(code) if target == "Python" else check_cpp(code)
