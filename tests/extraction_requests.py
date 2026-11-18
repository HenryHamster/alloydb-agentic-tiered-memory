"""Reads the conversation lines of an extraction request back, for fake Gemini clients."""

import re

_LINE = re.compile(r"\[(\d+)\] (?:(.+) \()?(user|assistant|tool)\)?: (.*)", re.S)


def sent_chunks(contents):
    """The chunks a request sent, as dicts of chunk, speaker, name and text."""
    source = contents.split("SOURCE:\n", 1)[1]
    chunks = []
    for line in source.split("\n"):
        if line.startswith("    ") and chunks:
            chunks[-1]["text"] += "\n" + line[4:]
            continue
        match = _LINE.fullmatch(line)
        if match:
            index, name, speaker, text = match.groups()
            chunks.append({"chunk": int(index), "speaker": speaker, "name": name, "text": text})
    return chunks
