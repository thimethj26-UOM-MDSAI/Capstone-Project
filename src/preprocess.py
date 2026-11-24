"""Text cleaning. Each step is a flag so it can be ablated (e.g. keep_emoji)."""
import re
import unicodedata

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
# broad emoji/symbol ranges
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️‍]")
ZWJ = "‍"  # Sinhala conjuncts use ZWJ (e.g. rakaransaya); handled below


def clean(text: str, keep_emoji: bool = False, keep_mentions: bool = False) -> str:
    t = unicodedata.normalize("NFC", text)
    t = URL.sub(" ", t)
    if not keep_mentions:
        t = MENTION.sub(" ", t)
    if not keep_emoji:
        # strip emoji but preserve ZWJ inside Sinhala (it joins consonant clusters)
        t = EMOJI.sub(lambda m: m.group(0) if m.group(0) == ZWJ else " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
