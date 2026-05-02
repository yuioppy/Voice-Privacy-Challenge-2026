"""
Compatibility patch for torchaudio 2.8+ with speechbrain 1.0.3
Some torchaudio versions removed list_audio_backends() which speechbrain still uses
"""
import torchaudio
import torch
import soundfile as sf


def _soundfile_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                    channels_first=True, format=None, buffer_size=4096,
                    backend=None):
    dtype = 'float32' if normalize else None
    frames = num_frames if num_frames is not None and num_frames > 0 else -1
    waveform, sample_rate = sf.read(
        uri,
        start=frame_offset,
        frames=frames,
        dtype=dtype,
        always_2d=True,
    )
    waveform = torch.from_numpy(waveform)
    if channels_first:
        waveform = waveform.transpose(0, 1)
    return waveform, sample_rate

# Patch torchaudio.list_audio_backends if it doesn't exist
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        """
        Compatibility function for torchaudio versions without list_audio_backends()
        Returns a list of available backends (newer versions handle backends automatically)
        """
        return ['soundfile', 'sox', 'ffmpeg']
    
    torchaudio.list_audio_backends = list_audio_backends

try:
    import torchcodec  # noqa: F401
except ImportError:
    torchaudio.load = _soundfile_load
