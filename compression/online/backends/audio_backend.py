"""AudioBackend: lossless bGPT audio compression (aligned with the team).

The medium is pre-split into fixed-duration 8 kHz / mono / 8-bit WAV chunks
(see utils.audio_utils.chunk_pydub_audio); each chunk is one byte-blob.  All
machinery is inherited from _BGPTByteBackend — only the extension and identity
differ.
"""
from __future__ import annotations

from compression.online.backends.bgpt_byte_backend import _BGPTByteBackend


class AudioBackend(_BGPTByteBackend):

    ext = "wav"

    @property
    def modality(self) -> str:
        return "audio"
