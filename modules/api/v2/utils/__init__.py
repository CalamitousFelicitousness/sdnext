"""
API v2 Utilities Package

Groups commonly used helpers for the v2 HTTP API.
This package intentionally avoids Gradio dependencies.
"""

# file utilities
from .file_utils import (
    FormFile,
    build_download_headers,
    build_multipart,
    create_temp_file,
    decode_base64_to_bytes,
    decode_base64_to_file,
    encode_bytes_to_base64,
    encode_file_to_base64,
    get_file_hash,
    guess_mime_type,
    handle_upload,
    iter_file,
    parse_multipart,
    read_bytes,
    sanitize_filename,
    write_bytes,
)

__all__ = [
    # file_utils
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
