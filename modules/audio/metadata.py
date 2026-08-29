"""Generation parameters embedded in audio files, the audio counterpart of png infotext.

Containers disagree about where tags live and which ones survive, so the reader is part of the
feature rather than an afterthought: ogg and opus report tags on the stream and nothing on the
container, and a container-only read misses them completely.
"""

import json
import os
from modules import shared
from modules.audio.stream import get_av
from modules.logger import log


INFO_KEY = 'parameters' # same key the png tEXt chunk uses, so one name covers both modalities


def create_audio_metadata(p=None, metadata: dict | None = None, filename: str | None = None) -> dict:
    """Container tags for one generation. `comment` duplicates the infotext so ordinary players
    show it; `description` is deliberately absent because ffmpeg maps comment onto it in flac and
    the two would overwrite each other."""
    metadata = metadata.copy() if metadata is not None else {}
    if not shared.opts.image_metadata:
        return metadata
    if p is None:
        return metadata
    from modules import processing
    try:
        info = processing.create_infotext(p)
    except Exception as e:
        log.debug(f'Audio metadata: infotext failed: {e}')
        info = ''
    if len(info) == 0:
        info = getattr(p, 'prompt', '') or ''
    if len(info) == 0:
        return metadata
    metadata.setdefault('title', os.path.basename(filename) if filename else 'SD.Next audio')
    metadata.setdefault('encoder', 'SD.Next')
    metadata.setdefault(INFO_KEY, info)
    metadata.setdefault('comment', info)
    return metadata


def read_audio_metadata(fn: str) -> dict:
    """Tags from the container and every audio stream, merged, keys lower-cased. ffmpeg's tag
    dictionary is case-insensitive on write, so the case a file comes back with is not the case
    it went in with."""
    av = get_av()
    if av is None:
        return {}
    tags = {}
    try:
        with av.open(fn) as container:
            sources = [container.metadata] + [stream.metadata for stream in container.streams.audio]
            for source in sources:
                for key, value in (source or {}).items():
                    tags.setdefault(str(key).lower(), value)
    except Exception as e:
        log.debug(f'Audio metadata read: file="{fn}" {e}')
    return tags


def read_audio_info(fn: str) -> str | None:
    """The embedded infotext, or None when the file carries none."""
    return read_audio_metadata(fn).get(INFO_KEY)


def read_sidecar(fn: str) -> dict | None:
    """The json written beside an audio file, which is the only parameter record for formats that
    cannot carry tags."""
    sidecar = f'{os.path.splitext(fn)[0]}.json'
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception as e:
        log.debug(f'Audio sidecar read: file="{sidecar}" {e}')
        return None


def write_sidecar(fn: str, metadata: dict | None = None, extra: dict | None = None) -> str | None:
    sidecar = f'{os.path.splitext(fn)[0]}.json'
    data = dict(metadata or {})
    if extra:
        data.update(extra)
    try:
        with open(sidecar, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return sidecar
    except Exception as e:
        log.error(f'Audio sidecar: file="{sidecar}" {e}')
        return None


def retag(fn: str, metadata: dict) -> bool:
    """Replace the tags on an existing file, copying packets verbatim so the audio is untouched."""
    av = get_av()
    if av is None:
        return False
    base, ext = os.path.splitext(fn)
    temp = f'{base}.retag{ext}'
    try:
        with av.open(fn) as src:
            if not src.streams.audio:
                return False
            source_stream = src.streams.audio[0]
            with av.open(temp, 'w') as dst:
                out_stream = dst.add_stream_from_template(source_stream)
                for key, value in metadata.items():
                    dst.metadata[key] = str(value)
                for packet in src.demux(source_stream):
                    if packet.dts is None: # the demuxer's flush packet carries no payload
                        continue
                    packet.stream = out_stream
                    dst.mux(packet)
        os.replace(temp, fn)
        return True
    except Exception as e:
        log.error(f'Audio retag: file="{fn}" {e}')
        if os.path.exists(temp):
            os.remove(temp)
        return False
