import numpy as np
import torch
from modules import errors
from modules.logger import log


# encoders that accept only a fixed set of rates; anything else has to be resampled on the way in
OPUS_RATES = (8000, 12000, 16000, 24000, 48000)
MP3_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
CODEC_RATES = {
    'libopus': OPUS_RATES,
    'opus': OPUS_RATES,
    'libmp3lame': MP3_RATES,
    'mp3': MP3_RATES,
}


def get_av():
    """The av module, or None when it is unavailable. Delegates so there is a single install path."""
    from modules.video_models.video_utils import check_av
    return check_av()


def get_audio_rate(p=None, default: int = 24000) -> int:
    # pipeline output wins when it reports a rate, else the loaded vocoder: LTX-2.0 runs at 24k,
    # 2.3 and 2.5 at 48k, and muxing at the wrong rate shifts the pitch
    from modules import shared
    rate = getattr(p, 'audio_sampling_rate', None) if p is not None else None
    if not rate:
        vocoder = getattr(shared.sd_model, 'vocoder', None)
        rate = getattr(getattr(vocoder, 'config', None), 'output_sampling_rate', None)
    return int(rate) if rate else default


def layout_name(channels: int) -> str:
    return 'stereo' if channels == 2 else 'mono'


def stream_rate(codec: str, sample_rate: int) -> int:
    """Rate the encoder will accept, snapping up to the nearest supported one when it is fussy."""
    rates = CODEC_RATES.get(codec)
    if not rates or sample_rate in rates:
        return int(sample_rate)
    higher = [r for r in rates if r >= sample_rate]
    return int(min(higher)) if higher else int(max(rates))


def normalize_waveform(audio) -> np.ndarray:
    """Waveform as float32 [channels, samples], from a tensor or array in either axis order."""
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio)
    if audio.ndim > 2:
        audio = np.squeeze(audio)
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim == 2 and audio.shape[0] > audio.shape[1] and audio.shape[1] in (1, 2):
        audio = audio.T
    return np.ascontiguousarray(audio.astype(np.float32, copy=False))


def is_decodable_audio(fn: str) -> bool:
    """Whether a file really holds audio, proven by decoding a frame of it.

    A stream listing is not proof: ffmpeg picks the demuxer from the filename, so opening an empty
    file named .flac succeeds and reports one audio stream. Only a decoded frame separates real
    audio from an extension.
    """
    av = get_av()
    if av is None:
        return False
    try:
        with av.open(fn) as container:
            if not container.streams.audio:
                return False
            for _frame in container.decode(audio=0):
                return True
    except Exception:
        return False
    return False


def add_audio_packets(container, audio_stream, audio: dict):
    if not audio or "frames" not in audio:
        return
    try:
        av = get_av()
        sr = audio.get("sr", 44100)
        layout = audio.get("layout", "stereo")
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=sr)
        fifo = av.AudioFifo()
        for raw_frame in audio.get("frames", []):
            for resampled in resampler.resample(raw_frame):
                fifo.write(resampled)
        for resampled in resampler.resample(None):
            fifo.write(resampled)
        pts_counter = 0
        frame_size = audio_stream.codec_context.frame_size or 1024
        while fifo.samples >= frame_size:
            frame = fifo.read(frame_size)
            frame.pts = pts_counter
            pts_counter += frame.samples
            for packet in audio_stream.encode(frame):
                packet.stream = audio_stream
                container.mux_one(packet)
        if fifo.samples > 0:
            frame = fifo.read(fifo.samples)
            frame.pts = pts_counter
            pts_counter += frame.samples
            for packet in audio_stream.encode(frame):
                packet.stream = audio_stream
                container.mux_one(packet)
        for packet in audio_stream.encode():
            packet.stream = audio_stream
            container.mux_one(packet)
    except Exception as e:
        log.error(f"Audio encode: type=packets {e}")
        errors.display(e, "Audio")


def waveform_channels(audio) -> int:
    """Channel count the encoder will be handed, after the normalization the writer applies.

    The stream has to be declared with this layout: declaring stereo for a mono waveform makes
    ffmpeg upmix it, and its power compensation drops the level by 3 dB.
    """
    samples = normalize_waveform(audio)
    return samples.shape[0] if samples.shape[0] in (1, 2) else 1


def add_audio_tensor(container, audio_stream, audio: torch.Tensor, sample_rate: int, target_rate: int | None = None):
    """Encode a waveform into an open container. `target_rate` resamples on the way in, for
    encoders that reject the source rate."""
    av = get_av()
    audio = normalize_waveform(audio)
    channels = audio.shape[0] if audio.shape[0] in (1, 2) else 1
    layout = layout_name(channels)
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    audio_frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio), format="s16p", layout=layout)
    audio_frame.sample_rate = sample_rate
    add_audio_packets(container, audio_stream, {"sr": target_rate or sample_rate, "layout": layout, "frames": [audio_frame]})
