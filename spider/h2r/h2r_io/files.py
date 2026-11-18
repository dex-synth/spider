"""Hashes, JSON and path rules shared by every sealed package."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def content_hash(value):
    """Hash a JSON value independently of key order and whitespace."""
    return text_hash(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_hashes(root, exclude=()):
    """Relative POSIX path -> sha256 for every file under root, except the named manifests."""
    root = Path(root)
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file() and path.name not in exclude}


def member(root, name):
    """Resolve a package-relative path and refuse anything outside the package."""
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or path != root and root not in path.parents:
        raise ValueError(f"Path leaves its package: {name}")
    return path


def verify_files(root, files):
    for name, expected in files.items():
        path = member(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed file: {Path(root).name}/{name}")


def read_manifest(root, name, format, identity):
    """Load a manifest and check its format and identity; `identity` maps the manifest to its id value.

    Any malformed manifest raises ValueError, so callers need not know the manifest layout.
    """
    manifest = read_json(Path(root) / name)
    key = format.split("-")[1] + "_id"
    if not isinstance(manifest, dict) or manifest.get("format") != format:
        raise ValueError(f"Expected {format} in {Path(root).name}/{name}")
    with manifest_fields(f"{Path(root).name}/{name}"):
        expected = content_hash(identity(manifest))
    if expected != manifest.get(key):
        raise ValueError(f"Identity changed: {Path(root).name}/{name}")
    return manifest


@contextmanager
def manifest_fields(label):
    """Report a missing or mistyped manifest field as ValueError."""
    try:
        yield
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed {label}: {type(error).__name__}: {error}") from error


def seal(root, name, manifest, identity):
    """Write a manifest with its id; the id rule lives next to the matching reader."""
    key = manifest["format"].split("-")[1] + "_id"
    manifest = {**manifest, key: content_hash(identity(manifest))}
    write_json(Path(root) / name, manifest)
    return manifest
