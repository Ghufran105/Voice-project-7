import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import speech_recognition as sr

import helpers as h


# ---------- parse_response
def test_parse_normal():
    code, expl = h.parse_response("<code>\nprint('hi')\n</code>\n<explanation>\nPrints hi.\n</explanation>")
    assert code == "print('hi')" and expl == "Prints hi."


def test_parse_strips_markdown_fences_inside_tags():
    code, _ = h.parse_response("<code>\n```python\nx = 1\n```\n</code><explanation>e</explanation>")
    assert code == "x = 1"


def test_parse_fallback_fenced_without_tags():
    code, expl = h.parse_response("Here you go:\n```cpp\nint main(){}\n```\nDone")
    assert code == "int main(){}" and "Here you go" in expl


def test_parse_plain_text_only():
    code, expl = h.parse_response("Which language do you mean?")
    assert code == "" and expl == "Which language do you mean?"


# ---------- syntax checks
def test_python_ok_and_bad():
    assert h.check_python("print(1)")[0] is True
    ok, msg = h.check_python("def f(:\n  pass")
    assert ok is False and "line 1" in msg


def test_cpp_ok_and_bad():
    good = '#include <iostream>\nint main(){ std::cout << 1; return 0; }'
    assert h.check_cpp(good)[0] is True
    ok, msg = h.check_cpp("int main(){ return 0 }")
    assert ok is False and "main.cpp" in msg


def test_check_syntax_empty_and_dispatch():
    assert h.check_syntax("   ", "Python")[0] is None
    assert h.check_syntax("x=", "Python")[0] is False
    assert h.check_syntax("int main(){return 0;}", "C++")[0] is True


# ---------- audio
def _wav(subtype, channels=1, rate=16000):
    t = np.linspace(0, 0.5, int(rate * 0.5), endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    data = np.stack([sig] * channels, axis=1) if channels > 1 else sig
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="WAV", subtype=subtype)
    return buf.getvalue()


@pytest.mark.parametrize("subtype,channels", [("PCM_16", 1), ("FLOAT", 1), ("FLOAT", 2), ("PCM_24", 1)])
def test_normalize_makes_wav_readable(subtype, channels):
    fixed = h._normalize_wav(_wav(subtype, channels))
    with sr.AudioFile(io.BytesIO(fixed)) as src:
        audio = sr.Recognizer().record(src)
    assert audio.sample_rate == 16000 and len(audio.frame_data) > 1000


def test_transcribe_success(monkeypatch):
    seen = {}

    def fake(self, audio, language=None, **kw):
        seen["lang"] = language
        return "  اردو test  "

    monkeypatch.setattr(sr.Recognizer, "recognize_google", fake)
    assert h.transcribe_audio(_wav("FLOAT"), "ur-PK") == "اردو test"
    assert seen["lang"] == "ur-PK"


def test_transcribe_errors(monkeypatch):
    with pytest.raises(h.SpeechError):
        h.transcribe_audio(b"", "en-US")
    with pytest.raises(h.SpeechError):
        h.transcribe_audio(b"not a wav file", "en-US")

    def unknown(self, *a, **k):
        raise sr.UnknownValueError()

    monkeypatch.setattr(sr.Recognizer, "recognize_google", unknown)
    with pytest.raises(h.SpeechError, match="could not understand"):
        h.transcribe_audio(_wav("PCM_16"), "en-US")

    def down(self, *a, **k):
        raise sr.RequestError("offline")

    monkeypatch.setattr(sr.Recognizer, "recognize_google", down)
    with pytest.raises(h.SpeechError, match="not reachable"):
        h.transcribe_audio(_wav("PCM_16"), "en-US")


# ---------- generation with a fake Groq client
def _reply(txt):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=txt))])


class FakeClient:
    last_kwargs = None
    reply = "<code>\nprint('ok')\n</code><explanation>Prints ok.</explanation>"

    def __init__(self, **kw):
        self.init = kw
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        FakeClient.last_kwargs = kw
        return _reply(FakeClient.reply)


def test_generate_success_and_prompt_shape(monkeypatch):
    monkeypatch.setattr(h.groq, "Groq", FakeClient)
    hist = [{"user_message": "u1", "raw": "a1"}] * 5
    out = h.generate_code("loop banao", "Python", "English", "key", history=hist)
    assert out["code"] == "print('ok')" and out["explanation"] == "Prints ok."
    kw = FakeClient.last_kwargs
    assert kw["model"] == h.DEFAULT_MODEL
    assert kw["messages"][0] == {"role": "system", "content": h.SYSTEM_PROMPT}
    assert len(kw["messages"]) == 1 + h.MAX_HISTORY_TURNS * 2 + 1  # system + capped history + new
    assert kw["messages"][-1]["role"] == "user"
    assert "TARGET LANGUAGE: Python" in kw["messages"][-1]["content"]
    assert "loop banao" in kw["messages"][-1]["content"]
    assert kw["reasoning_effort"] == "low"  # gpt-oss default model


def test_reasoning_effort_only_for_gpt_oss(monkeypatch):
    monkeypatch.setattr(h.groq, "Groq", FakeClient)
    h.generate_code("x", "C++", "Urdu", "key", model="llama-3.3-70b-versatile")
    assert "reasoning_effort" not in FakeClient.last_kwargs
    assert "EXPLANATION LANGUAGE: Urdu" in FakeClient.last_kwargs["messages"][-1]["content"]


def test_empty_model_answer(monkeypatch):
    monkeypatch.setattr(h.groq, "Groq", FakeClient)
    monkeypatch.setattr(FakeClient, "reply", "   ")
    with pytest.raises(h.CodeGenError, match="empty answer"):
        h.generate_code("x", "Python", "English", "key")
    monkeypatch.setattr(FakeClient, "reply", None)
    with pytest.raises(h.CodeGenError, match="empty answer"):
        h.generate_code("x", "Python", "English", "key")


def test_generate_input_validation():
    with pytest.raises(h.CodeGenError, match="say or type"):
        h.generate_code("  ", "Python", "English", "key")
    with pytest.raises(h.CodeGenError, match="Groq API key"):
        h.generate_code("x", "Python", "English", "")
    with pytest.raises(h.CodeGenError, match="too long"):
        h.generate_code("x" * 2000, "Python", "English", "key")


def test_generate_maps_api_errors(monkeypatch):
    req = SimpleNamespace(method="POST", url="https://api.groq.com/openai/v1/chat/completions")

    def resp(code):  # minimal stand-in for an HTTP response (works across SDK versions)
        return SimpleNamespace(status_code=code, headers={}, request=req)

    cases = [
        (h.groq.AuthenticationError("x", response=resp(401), body=None), "rejected"),
        (h.groq.NotFoundError("x", response=resp(404), body=None), "not found"),
        (h.groq.RateLimitError("x", response=resp(429), body=None), "Rate limit"),
        (h.groq.APIConnectionError(request=req), "Could not reach"),
        (h.groq.InternalServerError("boom", response=resp(500), body=None), "500"),
    ]
    for exc, expected in cases:
        class Boom(FakeClient):
            def _create(self, **kw):
                raise exc

        monkeypatch.setattr(h.groq, "Groq", Boom)
        with pytest.raises(h.CodeGenError, match=expected):
            h.generate_code("x", "Python", "English", "key")
