"""
File utilities for API v2

This module provides async-aware file operations to replace
Gradio dependencies, particularly gradio_client.utils functions.
"""

import base64
import tempfile
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Union, Tuple
import aiofiles
import asyncio
from io import BytesIO


async def decode_base64_to_file(
    base64_str: str,
    file_path: Optional[Path] = None,
    prefix: str = "sdnext_",
    suffix: str = ""
) -> Path:
    """
    Decode base64 string to file asynchronously.
    
    Replacement for gradio_client.utils.decode_base64_to_file
    
    Args:
        base64_str: Base64 encoded string
        file_path: Optional target file path
        prefix: Prefix for temporary file name
        suffix: Suffix for temporary file name
        
    Returns:
        Path to the created file
        
    Raises:
        ValueError: If base64 string is invalid
    """
    try:
        # Handle data URLs (e.g., "data:image/png;base64,...")
        if "," in base64_str and base64_str.startswith("data:"):
            base64_str = base64_str.split(",", 1)[1]
        
        # Decode base64
        data = base64.b64decode(base64_str, validate=True)
        
        # Create file path if not provided
        if file_path is None:
            fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
            file_path = Path(temp_path)
            # Close the file descriptor as aiofiles will open it
            import os
            os.close(fd)
        
        # Write data asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)
        
        return file_path
        
    except Exception as e:
        raise ValueError(f"Failed to decode base64 to file: {e}")


async def encode_file_to_base64(
    file_path: Union[str, Path],
    mime_type: Optional[str] = None
) -> str:
    """
    Encode file to base64 string asynchronously.
    
    Args:
        file_path: Path to the file
        mime_type: Optional MIME type for data URL
        
    Returns:
        Base64 encoded string, optionally as data URL
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    async with aiofiles.open(file_path, 'rb') as f:
        data = await f.read()
    
    base64_str = base64.b64encode(data).decode('utf-8')
    
    # Add data URL prefix if mime_type is provided
    if mime_type:
        return f"data:{mime_type};base64,{base64_str}"
    
    return base64_str


async def get_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "sha256"
) -> str:
    """
    Calculate file hash asynchronously.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
        
    Returns:
        Hex digest of the file hash
    """
    file_path = Path(file_path)
    hasher = hashlib.new(algorithm)
    
