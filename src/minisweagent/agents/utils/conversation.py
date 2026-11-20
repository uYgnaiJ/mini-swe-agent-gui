"""Helpers for distinguishing simple chat messages from software tasks.

The mini-SWE-agent is a tool-using software engineering agent. When a user
sends a greeting or another short social message, forcing the model through
the normal "call bash until you submit" loop produces noisy, pointless
commands. These helpers let the agent answer those messages immediately.
"""

from __future__ import annotations

import re
import unicodedata

# Patterns are matched against the *whole* normalized message (apart from
# trailing punctuation/emoji). This deliberately avoids treating a real task
# such as "hi, please fix the failing tests" as small talk.
_GREETING_PATTERNS = (
    r"hi+",
    r"hii+",
    r"hello+",
    r"hey+",
    r"hiya",
    r"howdy",
    r"yo",
    r"sup",
    r"what(?:'s| is) up",
    r"good (?:morning|afternoon|evening)",
    r"(?:hi|hello|hey) there",
    r"greetings",
    r"你好+",
    r"您好+",
    r"嗨+",
    r"哈喽+",
    r"喂",
    r"早上好",
    r"上午好",
    r"下午好",
    r"晚上好",
    r"在吗",
)

_THANKS_PATTERNS = (
    r"thanks?",
    r"thank you",
    r"thx",
    r"ty",
    r"谢谢+",
    r"感谢+",
)

_FAREWELL_PATTERNS = (
    r"bye+",
    r"goodbye",
    r"good bye",
    r"see you(?: later)?",
    r"再见",
    r"拜拜",
)

_IDENTITY_PATTERNS = (
    r"who are you",
    r"what are you",
    r"what can you do",
    r"what do you do",
    r"help",
    r"你是谁",
    r"你能做什么",
    r"你能干什么",
    r"你可以做什么",
    r"帮助",
)

_HOW_ARE_YOU_PATTERNS = (
    r"how are you",
    r"how(?:'s| is) it going",
    r"how are things",
    r"你好吗",
    r"最近怎么样",
)

_GREETING_REPLY = (
    "Hi! I'm mini-SWE-agent. I can help with software engineering tasks in this workspace. "
    "Tell me what you'd like me to investigate, fix, or test."
)
_THANKS_REPLY = "You're welcome! Let me know if you'd like me to look at or change something in the codebase."
_FAREWELL_REPLY = "Goodbye! Come back whenever you have a software engineering task for me."
_IDENTITY_REPLY = (
    "I'm mini-SWE-agent, a bash-using software engineering agent. I can inspect the repository, "
    "run tests, edit files, and verify fixes. Tell me what you'd like me to work on."
)
_HOW_ARE_YOU_REPLY = "I'm ready to help! Tell me what you'd like me to investigate, fix, or test in the workspace."

# A conversational message may contain only these characters around the
# phrase (punctuation, whitespace, and a few common emoji).
_TRAILING_NOISE_RE = re.compile(r"^[\s\W_]+|[\s\W_]+$", re.UNICODE)


def _normalize(message: str) -> str:
    text = unicodedata.normalize("NFKC", message).strip().casefold()
    text = _TRAILING_NOISE_RE.sub("", text)
    return re.sub(r"\s+", " ", text)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def conversational_reply(message: str) -> str | None:
    """Return a short reply when *message* is only social conversation.

    ``None`` means the message should be handled by the normal software
    engineering agent. The check is intentionally conservative: anything that
    contains an actual request stays on the normal agent path.
    """

    text = _normalize(message)
    if not text:
        return None
    if _matches(text, _HOW_ARE_YOU_PATTERNS):
        return _HOW_ARE_YOU_REPLY
    if _matches(text, _IDENTITY_PATTERNS):
        return _IDENTITY_REPLY
    if _matches(text, _THANKS_PATTERNS):
        return _THANKS_REPLY
    if _matches(text, _FAREWELL_PATTERNS):
        return _FAREWELL_REPLY
    if _matches(text, _GREETING_PATTERNS):
        return _GREETING_REPLY
    return None
