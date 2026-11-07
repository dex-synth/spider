"""h2r-evaluation-v1: length-framed JSON over a persistent local socket."""

import json
import math
import struct

MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def send(stream, value):
    """Send a single finite JSON object, separate from logs."""
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Protocol message too large")
    stream.sendall(struct.pack("!I", len(payload)) + payload)


def _read(stream, size):
    result = bytearray()
    while len(result) < size:
        chunk = stream.recv(size - len(result))
        if not chunk:
            raise EOFError("Evaluator disconnected")
        result.extend(chunk)
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate protocol field: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Non-finite JSON: {value}")


def _float(value):
    result = float(value)
    return result if math.isfinite(result) else _constant(value)


def receive(stream):
    """Read exactly one message, rejecting ambiguous or invalid JSON."""
    (size,) = struct.unpack("!I", _read(stream, 4))
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid protocol message size")
    result = json.loads(
        _read(stream, size),
        object_pairs_hook=_object,
        parse_float=_float,
        parse_constant=_constant,
    )
    if not isinstance(result, dict):
        raise ValueError("Expected a protocol object")
    return result
