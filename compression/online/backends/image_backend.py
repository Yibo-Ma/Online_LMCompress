"""ImageBackend: lossless bGPT image compression (aligned with the team).

The medium is pre-split into 32x32 BMP patches (see utils.img_utils.patchify_image);
each patch is one byte-blob chunk.  All machinery is inherited from
_BGPTByteBackend — only the extension and identity differ.
"""
from __future__ import annotations

import torch

from compression.online.backends.bgpt_byte_backend import _BGPTByteBackend


class ImageBackend(_BGPTByteBackend):

    ext = "bmp"

    @property
    def modality(self) -> str:
        return "image"
