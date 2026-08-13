import re

type JsonScalar = None | bool | int | float | str

PAIR_PATTERN = (
    r'\s*("(?:[^"\\]|\\.)*")\s*:\s*'
    + r'("(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?'
    + r"(?:[eE][+-]?\d+)?|NaN|-?Infinity|true|false|null)\s*"
)
PAIR = re.compile(PAIR_PATTERN)


def decode_json_string(token: str) -> str:
    # json.dumps emits escaped strings; decode only the JSON string grammar here.
    inner = token[1:-1]
    return bytes(inner, "utf-8").decode("unicode_escape")


def decode_json_scalar(token: str) -> JsonScalar:
    if token.startswith('"'):
        return decode_json_string(token)
    if token == "true":
        return True
    if token == "false":
        return False
    if token == "null":
        return None
    if token == "NaN":
        return float("nan")
    if token == "Infinity":
        return float("inf")
    if token == "-Infinity":
        return float("-inf")
    return float(token) if any(marker in token for marker in ".eE") else int(token)


def parse_flat_json_object(text: str) -> dict[str, JsonScalar]:
    position = 0
    source = text.strip()
    if not source.startswith("{") or not source.endswith("}"):
        raise ValueError("Checkpoint row must be a JSON object")
    position = 1
    end = len(source) - 1
    parsed: dict[str, JsonScalar] = {}
    while position < end:
        match = PAIR.match(source, position)
        if match is None:
            raise ValueError("Invalid checkpoint JSON")
        key_token, value_token = match.groups()
        parsed[decode_json_string(key_token)] = decode_json_scalar(value_token)
        position = match.end()
        if position < end:
            if source[position] != ",":
                raise ValueError("Invalid checkpoint JSON separator")
            position += 1
    return parsed
