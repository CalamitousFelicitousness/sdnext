"""Writing generated audio to disk.

flac is the default because it is the only container that is lossless, holds arbitrary tags
verbatim, and encodes fast. wav is offered but cannot carry parameters at all: ffmpeg's RIFF INFO
writer takes a fixed key whitelist and drops everything else without reporting it, which is why a
json sidecar is always written next to the audio rather than instead of the tags.
"""

import os
import time
from dataclasses import dataclass
import numpy as np
from modules import shared, errors, timer
from modules.audio import metadata as audio_metadata
from modules.audio.stream import add_audio_tensor, get_av, layout_name, normalize_waveform, stream_rate
from modules.logger import log


@dataclass
class AudioFormat:
    codec: str
    muxer: str
    tags: bool # whether the container carries arbitrary keys through ffmpeg
    lossless: bool


AUDIO_FORMATS = {
    'flac': AudioFormat(codec='flac', muxer='flac', tags=True, lossless=True),
    'opus': AudioFormat(codec='libopus', muxer='opus', tags=True, lossless=False),
    'mp3': AudioFormat(codec='libmp3lame', muxer='mp3', tags=True, lossless=False),
    'wav': AudioFormat(codec='pcm_s16le', muxer='wav', tags=False, lossless=True),
}
DEFAULT_FORMAT = 'flac'


def get_audio_filename(p=None) -> str:
    from modules.image.namegen import FilenameGenerator
    from modules.paths import resolve_output_path
    namegen = FilenameGenerator(p, seed=p.seed if p is not None else 0, prompt=p.prompt if p is not None else '')
    filename = namegen.apply(shared.opts.samples_filename_pattern if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0 else "[seq]-[prompt_words]")
    base_path = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_audio)
    if shared.opts.save_to_dirs:
        dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]")
        dirname = os.path.join(base_path, dirname, os.path.dirname(filename))
    else:
        dirname = base_path
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, filename)
    filename = namegen.sequence(filename)
    filename = namegen.sanitize(filename)
    return filename


def write_audio(fn: str, waveform, sample_rate: int, ext: str = DEFAULT_FORMAT, metadata: dict | None = None) -> str | None:
    """Encode one waveform to a container. Tags have to be set before the first packet is muxed,
    since anything written to the dictionary afterwards is dropped without an error."""
    av = get_av()
    if av is None:
        log.error('Audio: ffmpeg/av not available')
        return None
    spec = AUDIO_FORMATS.get(ext)
    if spec is None:
        log.error(f'Audio: file="{fn}" unknown format={ext} known={list(AUDIO_FORMATS)}')
        return None
    samples = normalize_waveform(waveform)
    if samples.shape[0] > 2:
        log.warning(f'Audio: channels={samples.shape[0]} keeping the first two')
        samples = samples[:2]
    channels = samples.shape[0]
    rate = stream_rate(spec.codec, sample_rate)
    if metadata and not spec.tags:
        log.info(f'Audio: format={ext} does not store parameters, the json sidecar carries them')
    try:
        with av.open(fn, mode='w', format=spec.muxer) as container:
            for key, value in (metadata or {}).items():
                container.metadata[key] = str(value)
            stream = container.add_stream(spec.codec, rate=rate)
            stream.layout = layout_name(channels)
            add_audio_tensor(container, stream, samples, sample_rate, target_rate=rate)
    except Exception as e:
        log.error(f'Audio encode: file="{fn}" codec={spec.codec} rate={rate} {e}')
        errors.display(e, 'Audio')
        return None
    return fn


def save_audio(
    p=None,
    waveform=None,
    sample_rate: int = 48000,
    ext: str = DEFAULT_FORMAT,
    metadata: dict | None = None,
    sidecar: bool = True,
    levels: dict | None = None,
    filename: str | None = None,
) -> str | None:
    """Write one generated waveform plus its parameter record. Returns the audio path, or None."""
    if waveform is None:
        return None
    t0 = time.time()
    ext = ext if ext in AUDIO_FORMATS else DEFAULT_FORMAT
    base = filename or get_audio_filename(p)
    base = os.path.splitext(base)[0]
    fn = f'{base}.{ext}'
    tags = audio_metadata.create_audio_metadata(p, metadata, fn)
    savejob = shared.state.begin('Save audio')
    try:
        written = write_audio(fn, waveform, sample_rate, ext=ext, metadata=tags)
        if written is None:
            return None
        if sidecar:
            audio_metadata.write_sidecar(fn, tags, extra={'levels': levels} if levels else None)
        size = os.path.getsize(fn) if os.path.exists(fn) else 0
        samples = normalize_waveform(waveform)
        duration = round(samples.shape[-1] / sample_rate, 2) if sample_rate else 0
        log.info(f'Audio: file="{fn}" format={ext} duration={duration} rate={sample_rate} channels={samples.shape[0]} size={size}')
        shared.state.outputs(fn)
    finally:
        shared.state.end(savejob)
    timer.process.add('save', time.time() - t0)
    return fn


def read_audio(fn: str, rate: int | None = None, layout: str | None = None) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Decode a file back to a float waveform and its rate.

    Passing rate or layout converts during the decode, which is what a model expecting 48 kHz
    stereo needs from a file that is neither: handing it the samples unconverted is a pitch shift.
    """
    av = get_av()
    if av is None:
        return None, None
    try:
        with av.open(fn) as container:
            if not container.streams.audio:
                return None, None
            stream = container.streams.audio[0]
            rate = int(rate or stream.codec_context.sample_rate)
            resampler = av.AudioResampler(format='fltp', layout=layout or stream.layout.name, rate=rate)
            chunks = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray())
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray())
    except Exception as e:
        log.error(f'Audio read: file="{fn}" {e}')
        return None, None
    if not chunks:
        return None, None
    return np.concatenate(chunks, axis=-1), rate
