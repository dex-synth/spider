"""The evaluation wire protocol: length-framed JSON over a local socket.

The evaluation server and controllers share this one implementation. It needs only the standard
library (Python >= 3.10) and nothing else from this package, so a controller may also use this
file on its own.

A message is a 4-byte big-endian length followed by that many bytes of UTF-8 JSON (one object).
The server sends `hello`, the controller answers with `start`, then `step` requests are answered
with `frames` until every episode is done. A rejected request is answered with
`{"type": "error", "message": ...}` and changes nothing.
"""

import json
import math
import struct

PROTOCOL = "h2r-evaluation-v2"
MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def encode(value):
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Protocol message too large")
    return payload


def send(stream, value):
    """Send a message object, or bytes already produced by encode()."""
    payload = value if isinstance(value, bytes) else encode(value)
    stream.sendall(struct.pack("!I", len(payload)) + payload)


def _read(stream, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.recv(count - len(chunks))
        if not chunk:
            raise EOFError("Connection closed by the peer")
        chunks.extend(chunk)
    return chunks


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate protocol field: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        return _constant(value)
    return result


def receive(stream):
    """Read one message; duplicate fields and non-finite numbers are rejected."""
    size, = struct.unpack("!I", _read(stream, 4))
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid protocol message size")
    value = json.loads(_read(stream, size), object_pairs_hook=_object,
                       parse_constant=_constant, parse_float=_float)
    if not isinstance(value, dict):
        raise ValueError("Protocol message must be an object")
    return value


def control_ticks(hello, control_dt_s):
    """Physics steps per control period; the period must be a whole number of `hello["dt_s"]`."""
    dt = hello["dt_s"]
    ticks = round(control_dt_s / dt)
    if ticks < 1 or not math.isclose(ticks * dt, control_dt_s, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"A control period of {control_dt_s:g} s is not a whole number of {dt:g} s physics steps")
    return ticks


def start_request(hello, *, method, controller, initial_robot, control_dt_s, bodies=(), contacts=False):
    """The `start` request for the task announced by `hello`.

    `method` is the method identity and `controller` free-form controller information (for example
    its checkpoint); both are recorded with the rollout. `initial_robot` maps every episode_id of
    `hello["episodes"]` to that episode's initial pose, as named joint position targets.
    `control_dt_s` is the controller's own period. Frames carry only the requested `bodies`, and
    contact records only when `contacts` is set.
    """
    control_ticks(hello, control_dt_s)
    return {"type": "start", "task_id": hello["task_id"],
            "binding": {"demo_id": hello["demo_id"], "robot_id": hello["robot_id"],
                        "method": method, "controller": controller},
            "requirements": {"bodies": list(bodies), "contacts": contacts, "control_dt_s": control_dt_s},
            "initial_robot": initial_robot}
