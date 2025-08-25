"""
File utilities for API v2

This module provides async-aware file operations that replace
Gradio dependencies (e.g., gradio_client.utils). Utilities are
kept UI-agnostic and safe for FastAPI/Starlette or any ASGI server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from pathlib import Path
import asyncio
import base64
import hashlib
import mimetypes
import os
import re
import secrets
import tempfile

import aiofiles

# --- small helpers ---------------------------------------------------------

_DATA_URL_PREFIX = re.compile(r"^data:.*;base64,", re.IGNORECASE)


def sanitize_filename(name: str, *, allow_unicode: bool = False) -> str:
    """Prevent path traversal & normalize unsafe characters (why: security)."""
    name = os.path.basename(name or "")
    if not allow_unicode:
        name = name.encode("ascii", "ignore").decode("ascii")
    # keep alnum, dash, underscore, dot; collapse others
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or f"file_{secrets.token_hex(4)}"


def guess_mime_type(
    path_or_name: Union[str, Path], default: str = "application/octet-stream"
) -> str:
    """Best-effort MIME detection using stdlib only (why: no heavy deps)."""
    mtype, _ = mimetypes.guess_type(str(path_or_name))
    return mtype or default


def create_temp_file(
    *, prefix: str = "sdnext_", suffix: str = "", dir: Optional[Union[str, Path]] = None
) -> Path:
    """Create a closed temp file and return its Path (why: aiofiles needs to open)."""
    fd, temp_path = tempfile.mkstemp(
        prefix=prefix, suffix=suffix, dir=str(dir) if dir else None
    )
    os.close(fd)
    return Path(temp_path)


def _strip_data_url_prefix(s: str) -> str:
    if _DATA_URL_PREFIX.match(s):
        return s.split(",", 1)[1]
    return s


def decode_base64_to_bytes(base64_str: str) -> bytes:
    """Decode base64 (supports data URLs) with validation."""
    s = _strip_data_url_prefix(base64_str).strip()
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        # tolerate missing padding
        pad = (-len(s)) % 4
        if pad:
            s += "=" * pad
        return base64.b64decode(s, validate=False)


def encode_bytes_to_base64(
    data: bytes, *, mime_type: Optional[str] = None, as_data_url: bool = False
) -> str:
    """Encode bytes to base64; optionally as data URL."""
    b64 = base64.b64encode(data).decode("utf-8")
    if as_data_url or mime_type:
        mt = mime_type or "application/octet-stream"
        return f"data:{mt};base64,{b64}"
    return b64


async def write_bytes(
    data: bytes, dest: Union[str, Path], *, overwrite: bool = False
) -> int:
    """Write bytes asynchronously; create parents; guard overwrite."""
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)
    return len(data)


async def read_bytes(path: Union[str, Path]) -> bytes:
    path = Path(path)
    async with aiofiles.open(path, "rb") as f:
        return await f.read()


async def iter_file(
    path: Union[str, Path], *, chunk_size: int = 65536
) -> AsyncIterator[bytes]:
    """Async file iterator (why: efficient http streaming)."""
    path = Path(path)
    async with aiofiles.open(path, "rb") as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            yield chunk


# --- replacements for gradio utils -----------------------------------------


async def decode_base64_to_file(
    base64_str: str,
    file_path: Optional[Path] = None,
    prefix: str = "sdnext_",
    suffix: str = "",
) -> Path:
    """
    Decode base64 string to file (replacement for gradio_client.utils.decode_base64_to_file).
    """
    data = decode_base64_to_bytes(base64_str)
    if file_path is None:
        file_path = create_temp_file(prefix=prefix, suffix=suffix)
    await write_bytes(data, file_path, overwrite=True)
    return file_path


async def encode_file_to_base64(
    file_path: Union[str, Path],
    mime_type: Optional[str] = None,
    *,
    as_data_url: bool = False,
) -> str:
    """
    Encode file to base64 (optionally as data URL).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    data = await read_bytes(file_path)
    mime = mime_type or guess_mime_type(file_path)
    return encode_bytes_to_base64(data, mime_type=mime, as_data_url=as_data_url)


async def get_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "sha256",
) -> str:
    """
    Calculate file hash asynchronously.
    """
    file_path = Path(file_path)
    hasher = hashlib.new(algorithm)
    async with aiofiles.open(file_path, "rb") as f:
        while True:
            chunk = await f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


# --- upload/download handlers ----------------------------------------------


def build_download_headers(
    path: Union[str, Path],
    *,
    filename: Optional[str] = None,
    inline: bool = False,
    mime_type: Optional[str] = None,
) -> Dict[str, str]:
    """
    Prepare common headers for HTTP file responses (UI-agnostic).
    """
    p = Path(path)
    fname = sanitize_filename(filename or p.name)
    mtype = mime_type or guess_mime_type(fname)
    size = p.stat().st_size if p.exists() else 0
    # RFC 6266 filename* for UTF-8; also include plain filename fallback
    dispo = "inline" if inline else "attachment"
    headers = {
        "Content-Type": mtype,
        "Content-Length": str(size),
        "Content-Disposition": f"{dispo}; filename=\"{fname}\"; filename*=UTF-8''{fname}",
        "Accept-Ranges": "bytes",
    }
    return headers


async def _read_all(part: Any) -> Tuple[bytes, Optional[str]]:
    """
    Duck-type part reader:
    - Starlette/FastAPI UploadFile: async .read(), .filename, .content_type
    - aiohttp multipart: .read(), .filename, .headers.get("Content-Type")
    - Our FormFile: .data, .content_type
    - (filename, bytes, [content_type])
    - bytes
    """
    # Our FormFile
    if isinstance(part, FormFile):
        return part.data or b"", part.content_type
    # tuple variants
    if isinstance(part, tuple):
        if len(part) == 2:
            _, data = part
            return (data or b""), None
        if len(part) >= 3:
            _, data, ctype = part[:3]
            return (data or b""), ctype
    # raw bytes
    if isinstance(part, (bytes, bytearray, memoryview)):
        b = bytes(part)
        return b, None
    # UploadFile-like
    data = None
    ctype = getattr(part, "content_type", None) or getattr(
        getattr(part, "headers", {}), "get", lambda *_: None
    )("Content-Type")
    read_fn = getattr(part, "read", None)
    if asyncio.iscoroutinefunction(read_fn):
        data = await read_fn()
    elif callable(read_fn):
        data = read_fn()
    elif hasattr(part, "file"):  # Starlette UploadFile .file (SpooledTemporaryFile)
        fobj = part.file
        data = fobj.read() if callable(getattr(fobj, "read", None)) else None
    if data is None:
        raise TypeError("Unsupported upload part type")
    return data, ctype


async def handle_upload(
    part: Any,
    dest_dir: Union[str, Path],
    *,
    filename: Optional[str] = None,
    max_size: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """
    Save an uploaded part to dest_dir.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # resolve target filename
    inferred_name = (
        getattr(part, "filename", None)
        or (part[0] if isinstance(part, tuple) and part else None)
        or filename
        or f"upload_{secrets.token_hex(4)}"
    )
    safe_name = sanitize_filename(str(inferred_name))
    target = dest / safe_name
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")

    data, _ctype = await _read_all(part)
    if max_size is not None and len(data) > max_size:
        raise ValueError(f"Uploaded file exceeds max_size={max_size} bytes")

    await write_bytes(data, target, overwrite=True)
    return target


# --- multipart form-data ----------------------------------------------------


@dataclass
class FormFile:
    name: str
    filename: str
    content_type: str
    data: bytes

    async def save(self, path: Union[str, Path], *, overwrite: bool = False) -> Path:
        """Persist to disk."""
        p = Path(path)
        await write_bytes(self.data, p, overwrite=overwrite)
        return p


def _extract_disposition_params(header: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in re.finditer(r';\s*([a-zA-Z0-9_-]+)\s*=\s*"([^"]*)"', header):
        out[m.group(1).lower()] = m.group(2)
    return out


def _ensure_bytes(x: Union[str, bytes]) -> bytes:
    return (
        x if isinstance(x, (bytes, bytearray, memoryview)) else str(x).encode("utf-8")
    )


def build_multipart(
    fields: Mapping[str, Union[str, bytes, Tuple[str, bytes], Tuple[str, bytes, str]]],
    *,
    boundary: Optional[str] = None,
) -> Tuple[bytes, str]:
    """
    Build multipart/form-data body from mapping.

    Value options:
      - "text"
      - b"bytes"
      - (filename, bytes)
      - (filename, bytes, content_type)
    """
    boundary = boundary or f"----sdnext{secrets.token_hex(12)}"
    lines: list[bytes] = []
    b = boundary
    for name, value in fields.items():
        name_b = _ensure_bytes(name)
        lines.append(f"--{b}\r\n".encode("ascii"))
        if isinstance(value, tuple):
            if len(value) == 2:
                filename, data = value
                ctype = guess_mime_type(filename)
            else:
                filename, data, ctype = value[:3]
            fname = sanitize_filename(str(filename))
            dispo = f'form-data; name="{name}"; filename="{fname}"\r\n'
            lines.append(f"Content-Disposition: {dispo}".encode("utf-8"))
            lines.append(f"Content-Type: {ctype}\r\n\r\n".encode("utf-8"))
            lines.append(_ensure_bytes(data))
            lines.append(b"\r\n")
        else:
            dispo = f'form-data; name="{name}"\r\n'
            lines.append(f"Content-Disposition: {dispo}".encode("utf-8"))
            lines.append(b"Content-Type: text/plain; charset=utf-8\r\n\r\n")
            lines.append(_ensure_bytes(value))
            lines.append(b"\r\n")
    lines.append(f"--{b}--\r\n".encode("ascii"))
    body = b"".join(lines)
    ctype_header = f"multipart/form-data; boundary={boundary}"
    return body, ctype_header


def parse_multipart(
    data: bytes,
    content_type: str,
) -> Dict[str, Union[str, FormFile]]:
    """
    Minimal multipart/form-data parser (synchronous by design).
    - Parses small/medium payloads into memory.
    - Avoids external deps; suitable for utility use.
    """
    m = re.search(r"boundary=([-\w\'()+_.,:=?]+)", content_type, re.IGNORECASE)
    if not m:
        raise ValueError("Missing multipart boundary in Content-Type")
    boundary = m.group(1).strip('"')

    out: Dict[str, Union[str, FormFile]] = {}
    sep = ("--" + boundary).encode("ascii")
    end = ("--" + boundary + "--").encode("ascii")

    # split on boundaries, discard preamble/epilogue
    parts = data.split(sep)
    for part in parts:
        if not part or part.startswith(b"--") or part == b"\r\n":
            continue
        if part.endswith(b"\r\n"):
            part = part[:-2]
        # header / body split
        try:
            header_blob, body = part.split(b"\r\n\r\n", 1)
        except ValueError:
            continue
        headers = header_blob.decode("utf-8", "ignore").split("\r\n")
        disp = next(
            (h for h in headers if h.lower().startswith("content-disposition:")), ""
        )
        if not disp:
            continue
        params = _extract_disposition_params(disp)
        name = params.get("name") or f"part_{secrets.token_hex(3)}"
        filename = params.get("filename")
        ctype = "text/plain; charset=utf-8"
        for h in headers:
            if h.lower().startswith("content-type:"):
                ctype = h.split(":", 1)[1].strip()
                break
        if filename:
            out[name] = FormFile(
                name=name,
                filename=sanitize_filename(filename),
                content_type=ctype,
                data=body,
            )
        else:
            # text field
            try:
                out[name] = body.decode("utf-8")
            except UnicodeDecodeError:
                out[name] = body.decode("latin-1", "ignore")
    return out


# --- module __all__ ---------------------------------------------------------

__all__ = [
    "FormFile",
    "build_download_headers",
    "build_multipart",
    "create_temp_file",
    "decode_base64_to_bytes",
    "decode_base64_to_file",
    "encode_bytes_to_base64",
    "encode_file_to_base64",
    "get_file_hash",
    "guess_mime_type",
    "handle_upload",
    "iter_file",
    "parse_multipart",
    "read_bytes",
    "sanitize_filename",
    "write_bytes",
]
