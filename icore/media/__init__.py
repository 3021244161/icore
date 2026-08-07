"""
icore.media - Multimodal file abstractions and processors.

Provides:
    - ``MediaType`` / ``MediaSource``: Enumerations for media types and sources.
    - ``MediaFile``:                    Unified media file abstraction with
                                        lazy loading (local / URL / base64 / bytes).
    - ``BaseMediaProcessor``:           Abstract processor interface.
    - ``ImageProcessor``:               Pillow-backed image processor (lazy import).
    - ``VideoProcessor``:               ffmpeg-backed video processor (lazy import).
    - ``AudioProcessor``:               pydub/whisper-backed audio processor (lazy import).
    - ``MediaProcessorRegistry``:       Routes ``MediaFile`` -> matching processor.

Module dependency:
    media depends on icore.exceptions only. Heavy image / audio / video
    backends (Pillow, pydub, ffmpeg, whisper) are imported lazily so
    this package is importable without those installed.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from icore.exceptions import MediaUnsupportedError, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MediaType(str, Enum):
    """Media type classification."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"


class MediaSource(str, Enum):
    """Where the media content comes from."""

    LOCAL = "local"     # Filesystem path
    URL = "url"         # HTTP(S) URL
    BASE64 = "base64"   # Base64-encoded string
    BYTES = "bytes"     # In-memory bytes


# ---------------------------------------------------------------------------
# MIME / extension helpers
# ---------------------------------------------------------------------------

#: Maps (media_type, extension) for common formats.
_EXT_BY_TYPE: dict[MediaType, list[str]] = {
    MediaType.IMAGE: [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"],
    MediaType.VIDEO: [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"],
    MediaType.AUDIO: [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
    MediaType.PDF: [".pdf"],
}

#: Maps (media_type, mime) defaults.
_MIME_BY_TYPE: dict[MediaType, str] = {
    MediaType.IMAGE: "image/jpeg",
    MediaType.VIDEO: "video/mp4",
    MediaType.AUDIO: "audio/mpeg",
    MediaType.PDF: "application/pdf",
}


# ---------------------------------------------------------------------------
# MediaFile
# ---------------------------------------------------------------------------


@dataclass
class MediaFile:
    """
    Multimodal file abstraction.

    Any Task receives this uniform object regardless of whether the
    underlying bytes live on local disk, an HTTP URL, a base64 string
    or in-memory bytes. Decoding is lazy: the raw bytes / base64 are
    only resolved on first access.

    Supports context-manager usage to release cached bytes::

        async with media_file as mf:
            data = await mf.to_bytes()

    Attributes:
        media_type:   One of MediaType.
        source_type:  One of MediaSource.
        source:       Path / URL / base64 / raw bytes depending on source_type.
        metadata:     Free-form metadata dict.
    """

    media_type: MediaType
    source_type: MediaSource
    source: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    # Lazy caches.
    _raw_bytes: Optional[bytes] = field(default=None, repr=False)
    _base64: Optional[str] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Bytes / base64 access
    # ------------------------------------------------------------------

    async def to_bytes(self) -> bytes:
        """Return the raw bytes of this media file (lazy-load)."""
        if self._raw_bytes is not None:
            return self._raw_bytes

        if self.source_type == MediaSource.BYTES:
            if not isinstance(self.source, (bytes, bytearray)):
                raise ValidationError(
                    "MediaFile source_type=BYTES requires bytes source"
                )
            self._raw_bytes = bytes(self.source)
        elif self.source_type == MediaSource.BASE64:
            if not isinstance(self.source, str):
                raise ValidationError(
                    "MediaFile source_type=BASE64 requires str source"
                )
            self._raw_bytes = base64.b64decode(self.source)
        elif self.source_type == MediaSource.LOCAL:
            if not isinstance(self.source, str):
                raise ValidationError(
                    "MediaFile source_type=LOCAL requires path string"
                )
            self._raw_bytes = await asyncio.to_thread(self._read_local)
        elif self.source_type == MediaSource.URL:
            if not isinstance(self.source, str):
                raise ValidationError(
                    "MediaFile source_type=URL requires URL string"
                )
            self._raw_bytes = await self._read_url()
        else:
            raise MediaUnsupportedError(
                f"Unsupported media source: {self.source_type}"
            )
        return self._raw_bytes

    async def to_base64(self) -> str:
        """Return the base64-encoded content (lazy-load)."""
        if self._base64 is not None:
            return self._base64
        data = await self.to_bytes()
        self._base64 = base64.b64encode(data).decode("ascii")
        return self._base64

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def extension(self) -> str:
        """Best-effort file extension (including the leading dot)."""
        # 1. metadata override
        ext = self.metadata.get("extension")
        if ext:
            return ext if ext.startswith(".") else f".{ext}"
        # 2. local path / URL
        if isinstance(self.source, str):
            guessed = Path(self.source).suffix.lower()
            if guessed:
                return guessed
        # 3. default by media_type
        return _EXT_BY_TYPE.get(self.media_type, [".bin"])[0]

    @property
    def mime_type(self) -> str:
        """Best-effort MIME type."""
        mime = self.metadata.get("mime_type")
        if mime:
            return mime
        if isinstance(self.source, str):
            guessed, _ = mimetypes.guess_type(self.source)
            if guessed:
                return guessed
        return _MIME_BY_TYPE.get(self.media_type, "application/octet-stream")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_local(self) -> bytes:
        if not os.path.exists(self.source):
            raise ValidationError(f"Local media file not found: {self.source}")
        with open(self.source, "rb") as f:
            return f.read()

    async def _read_url(self) -> bytes:
        try:
            import httpx  # type: ignore
        except ImportError as e:
            raise ImportError(
                "httpx is required for MediaFile URL source. "
                "Install with: pip install httpx"
            ) from e
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(self.source)
            resp.raise_for_status()
            return resp.content

    # ------------------------------------------------------------------
    # Context manager: release cached bytes on exit
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MediaFile":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        """Drop cached bytes/base64 so memory is reclaimed."""
        self._raw_bytes = None
        self._base64 = None


# ---------------------------------------------------------------------------
# Processor interface
# ---------------------------------------------------------------------------


class BaseMediaProcessor(ABC):
    """Abstract media processor interface."""

    @abstractmethod
    async def extract_metadata(self, file: MediaFile) -> dict[str, Any]:
        """Extract metadata (resolution / duration / sample rate / EXIF)."""
        raise NotImplementedError

    @abstractmethod
    async def transcode(
        self, file: MediaFile, target_format: str
    ) -> MediaFile:
        """Transcode to a different format. Returns a new MediaFile."""
        raise NotImplementedError

    @abstractmethod
    async def generate_thumbnail(
        self,
        file: MediaFile,
        size: tuple[int, int] = (256, 256),
    ) -> MediaFile:
        """Generate a thumbnail image."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Image processor (Pillow, lazy)
# ---------------------------------------------------------------------------


class ImageProcessor(BaseMediaProcessor):
    """Image processor backed by Pillow (lazy import)."""

    async def extract_metadata(self, file: MediaFile) -> dict[str, Any]:
        try:
            from PIL import Image  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Pillow is required for ImageProcessor. "
                "Install with: pip install pillow"
            ) from e
        data = await file.to_bytes()

        def _extract() -> dict[str, Any]:
            with Image.open(__import__("io").BytesIO(data)) as img:
                meta: dict[str, Any] = {
                    "format": img.format,
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                }
                exif = img.getexif() if hasattr(img, "getexif") else None
                if exif:
                    meta["exif"] = dict(exif)
                return meta

        return await asyncio.to_thread(_extract)

    async def transcode(
        self, file: MediaFile, target_format: str
    ) -> MediaFile:
        try:
            from PIL import Image  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Pillow is required for ImageProcessor. "
                "Install with: pip install pillow"
            ) from e
        data = await file.to_bytes()
        fmt = target_format.upper().lstrip(".")

        def _transcode() -> bytes:
            import io

            with Image.open(io.BytesIO(data)) as img:
                out = io.BytesIO()
                img.save(out, format=fmt)
                return out.getvalue()

        new_bytes = await asyncio.to_thread(_transcode)
        return MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=new_bytes,
            metadata={
                **file.metadata,
                "mime_type": f"image/{target_format.lower()}",
                "extension": f".{target_format.lower()}",
            },
        )

    async def generate_thumbnail(
        self,
        file: MediaFile,
        size: tuple[int, int] = (256, 256),
    ) -> MediaFile:
        try:
            from PIL import Image  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Pillow is required for ImageProcessor. "
                "Install with: pip install pillow"
            ) from e
        data = await file.to_bytes()

        def _thumb() -> bytes:
            import io

            with Image.open(io.BytesIO(data)) as img:
                img.thumbnail(size)
                out = io.BytesIO()
                img.save(out, format="PNG")
                return out.getvalue()

        new_bytes = await asyncio.to_thread(_thumb)
        return MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=new_bytes,
            metadata={
                "mime_type": "image/png",
                "extension": ".png",
                "thumbnail_size": size,
            },
        )

    async def ocr(
        self, file: MediaFile, lang: str = "chi_sim"
    ) -> str:
        """
        OCR text recognition (best-effort).

        Tries pytesseract first; if not installed, raises a clear error.
        """
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pytesseract + Pillow required for OCR. "
                "Install with: pip install pytesseract pillow"
            ) from e
        data = await file.to_bytes()

        def _ocr() -> str:
            import io

            with Image.open(io.BytesIO(data)) as img:
                return pytesseract.image_to_string(img, lang=lang)

        return await asyncio.to_thread(_ocr)


# ---------------------------------------------------------------------------
# Video processor (ffmpeg, lazy)
# ---------------------------------------------------------------------------


class VideoProcessor(BaseMediaProcessor):
    """Video processor backed by ffmpeg (lazy subprocess import)."""

    async def extract_metadata(self, file: MediaFile) -> dict[str, Any]:
        # Use ffprobe if available; otherwise fall back to size only.
        if file.source_type == MediaSource.LOCAL and isinstance(file.source, str):
            return await self._ffprobe(file.source)
        # For non-local sources, only return size info.
        data = await file.to_bytes()
        return {"size_bytes": len(data)}

    async def _ffprobe(self, path: str) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v", "error",
                "-show_format",
                "-show_streams",
                "-of", "json",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0 or not stdout:
                return {"path": path}
            import json

            return json.loads(stdout.decode("utf-8", errors="replace"))
        except FileNotFoundError:
            return {"path": path, "note": "ffprobe not installed"}
        except Exception as e:
            return {"path": path, "error": str(e)}

    async def transcode(
        self, file: MediaFile, target_format: str
    ) -> MediaFile:
        # Best-effort: requires ffmpeg + local file.
        if file.source_type != MediaSource.LOCAL:
            raise ValidationError(
                "Video transcoding requires LOCAL source (write to disk first)"
            )
        src_path = file.source
        out_path = f"{src_path}.transcoded.{target_format}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i", src_path,
                "-c:v", "libx264" if target_format.lower() in ("mp4", "mkv") else "copy",
                "-c:a", "aac" if target_format.lower() in ("mp4", "mkv") else "copy",
                out_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                raise ValidationError(
                    f"ffmpeg transcoding failed (exit={proc.returncode})"
                )
        except FileNotFoundError as e:
            raise ImportError(
                "ffmpeg is required for video transcoding. "
                "Install ffmpeg and ensure it is on PATH."
            ) from e
        return MediaFile(
            media_type=MediaType.VIDEO,
            source_type=MediaSource.LOCAL,
            source=out_path,
            metadata={
                **file.metadata,
                "transcoded_from": src_path,
                "extension": f".{target_format}",
            },
        )

    async def generate_thumbnail(
        self,
        file: MediaFile,
        size: tuple[int, int] = (256, 256),
    ) -> MediaFile:
        if file.source_type != MediaSource.LOCAL:
            raise ValidationError(
                "Video thumbnail requires LOCAL source (write to disk first)"
            )
        out_path = f"{file.source}.thumb.jpg"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i", file.source,
                "-ss", "00:00:01",
                "-vframes", "1",
                "-vf", f"scale={size[0]}:{size[1]}",
                out_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                raise ValidationError(
                    f"ffmpeg thumbnail failed (exit={proc.returncode})"
                )
        except FileNotFoundError as e:
            raise ImportError(
                "ffmpeg is required for video thumbnails."
            ) from e
        return MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.LOCAL,
            source=out_path,
            metadata={
                "mime_type": "image/jpeg",
                "extension": ".jpg",
                "thumbnail_size": size,
            },
        )

    async def extract_keyframes(
        self,
        file: MediaFile,
        fps: int = 1,
    ) -> list[MediaFile]:
        """
        Extract keyframes at ``fps`` frames-per-second.

        Frames are written as ``<source>.frame.<n>.jpg``.
        """
        if file.source_type != MediaSource.LOCAL:
            raise ValidationError(
                "Keyframe extraction requires LOCAL source"
            )
        out_pattern = f"{file.source}.frame.%04d.jpg"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i", file.source,
                "-vf", f"fps={fps}",
                out_pattern,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                raise ValidationError(
                    f"ffmpeg keyframe extraction failed (exit={proc.returncode})"
                )
        except FileNotFoundError as e:
            raise ImportError(
                "ffmpeg is required for keyframe extraction."
            ) from e
        # Collect generated frame files.
        frames: list[MediaFile] = []
        parent = Path(file.source).parent
        prefix = Path(file.source).name + ".frame."
        for entry in sorted(parent.iterdir()):
            if entry.name.startswith(prefix) and entry.name.endswith(".jpg"):
                frames.append(
                    MediaFile(
                        media_type=MediaType.IMAGE,
                        source_type=MediaSource.LOCAL,
                        source=str(entry),
                        metadata={
                            "mime_type": "image/jpeg",
                            "extension": ".jpg",
                            "extracted_from": file.source,
                            "fps": fps,
                        },
                    )
                )
        return frames

    async def extract_audio_track(self, file: MediaFile) -> MediaFile:
        """Extract the audio track as a WAV file."""
        if file.source_type != MediaSource.LOCAL:
            raise ValidationError(
                "Audio extraction requires LOCAL source"
            )
        out_path = f"{file.source}.audio.wav"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i", file.source,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                out_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                raise ValidationError(
                    f"ffmpeg audio extraction failed (exit={proc.returncode})"
                )
        except FileNotFoundError as e:
            raise ImportError(
                "ffmpeg is required for audio extraction."
            ) from e
        return MediaFile(
            media_type=MediaType.AUDIO,
            source_type=MediaSource.LOCAL,
            source=out_path,
            metadata={
                "mime_type": "audio/wav",
                "extension": ".wav",
                "extracted_from": file.source,
            },
        )


# ---------------------------------------------------------------------------
# Audio processor (pydub, lazy)
# ---------------------------------------------------------------------------


class AudioProcessor(BaseMediaProcessor):
    """Audio processor backed by pydub (lazy import)."""

    async def extract_metadata(self, file: MediaFile) -> dict[str, Any]:
        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pydub is required for AudioProcessor. "
                "Install with: pip install pydub"
            ) from e
        data = await file.to_bytes()

        def _extract() -> dict[str, Any]:
            import io

            audio = AudioSegment.from_file(io.BytesIO(data))
            return {
                "channels": audio.channels,
                "sample_width": audio.sample_width,
                "frame_rate": audio.frame_rate,
                "duration_seconds": len(audio) / 1000.0,
            }

        return await asyncio.to_thread(_extract)

    async def transcode(
        self, file: MediaFile, target_format: str
    ) -> MediaFile:
        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pydub is required for AudioProcessor. "
                "Install with: pip install pydub"
            ) from e
        data = await file.to_bytes()
        fmt = target_format.lower().lstrip(".")

        def _transcode() -> bytes:
            import io

            audio = AudioSegment.from_file(io.BytesIO(data))
            out = io.BytesIO()
            audio.export(out, format=fmt)
            return out.getvalue()

        new_bytes = await asyncio.to_thread(_transcode)
        return MediaFile(
            media_type=MediaType.AUDIO,
            source_type=MediaSource.BYTES,
            source=new_bytes,
            metadata={
                **file.metadata,
                "extension": f".{fmt}",
            },
        )

    async def generate_thumbnail(
        self,
        file: MediaFile,
        size: tuple[int, int] = (256, 256),
    ) -> MediaFile:
        """Waveform image is non-trivial; return a placeholder text file."""
        raise MediaUnsupportedError(
            "Audio thumbnail (waveform) is not implemented; "
            "use a dedicated visualization library."
        )

    async def transcribe(
        self,
        file: MediaFile,
        language: str = "zh",
    ) -> str:
        """
        Speech-to-text via whisper (lazy import).

        Returns the transcribed text.
        """
        try:
            import whisper  # type: ignore
        except ImportError as e:
            raise ImportError(
                "openai-whisper is required for transcription. "
                "Install with: pip install openai-whisper"
            ) from e

        # Whisper needs a local file; write one if source is not local.
        if file.source_type == MediaSource.LOCAL and isinstance(file.source, str):
            local_path = file.source
        else:
            data = await file.to_bytes()
            import tempfile

            fd, local_path = tempfile.mkstemp(
                suffix=file.extension,
                prefix="icore_asr_",
            )
            os.close(fd)
            with open(local_path, "wb") as f:
                f.write(data)

        def _transcribe() -> str:
            model = whisper.load_model("base")
            result = model.transcribe(local_path, language=language)
            return result.get("text", "")

        return await asyncio.to_thread(_transcribe)

    async def split_by_silence(
        self,
        file: MediaFile,
        min_silence_len: int = 500,
    ) -> list[MediaFile]:
        """Split an audio file into chunks at silence boundaries."""
        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pydub is required for AudioProcessor. "
                "Install with: pip install pydub"
            ) from e
        data = await file.to_bytes()

        def _split() -> list[bytes]:
            import io

            audio = AudioSegment.from_file(io.BytesIO(data))
            chunks = AudioSegment.silent(
                duration=min_silence_len
            ).split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=audio.dBFS - 16,
            )
            out: list[bytes] = []
            for c in chunks or []:
                buf = io.BytesIO()
                c.export(buf, format="wav")
                out.append(buf.getvalue())
            return out

        results = await asyncio.to_thread(_split)
        return [
            MediaFile(
                media_type=MediaType.AUDIO,
                source_type=MediaSource.BYTES,
                source=chunk_bytes,
                metadata={
                    **file.metadata,
                    "mime_type": "audio/wav",
                    "extension": ".wav",
                    "split_index": i,
                },
            )
            for i, chunk_bytes in enumerate(results)
        ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MediaProcessorRegistry:
    """
    Routes a ``MediaFile`` to its matching processor.

    Usage::

        registry = MediaProcessorRegistry()
        registry.register(MediaType.IMAGE, ImageProcessor())
        processor = registry.get(media_file)
        meta = await processor.extract_metadata(media_file)
    """

    def __init__(self) -> None:
        self._processors: dict[MediaType, BaseMediaProcessor] = {}

    def register(
        self, media_type: MediaType, processor: BaseMediaProcessor
    ) -> None:
        """Register a processor for a MediaType."""
        self._processors[media_type] = processor

    def get(self, file: MediaFile) -> BaseMediaProcessor:
        """Return the processor registered for ``file.media_type``."""
        processor = self._processors.get(file.media_type)
        if processor is None:
            raise MediaUnsupportedError(
                f"No processor registered for media_type={file.media_type}"
            )
        return processor

    def has(self, media_type: MediaType) -> bool:
        return media_type in self._processors

    def list_types(self) -> list[MediaType]:
        return list(self._processors.keys())


def create_default_media_registry() -> MediaProcessorRegistry:
    """
    Build a registry pre-populated with the default processors.

    Heavy backends (Pillow / ffmpeg / pydub) are imported lazily by
    each processor, so this factory is cheap.
    """
    registry = MediaProcessorRegistry()
    registry.register(MediaType.IMAGE, ImageProcessor())
    registry.register(MediaType.VIDEO, VideoProcessor())
    registry.register(MediaType.AUDIO, AudioProcessor())
    return registry


__all__ = [
    "MediaType",
    "MediaSource",
    "MediaFile",
    "BaseMediaProcessor",
    "ImageProcessor",
    "VideoProcessor",
    "AudioProcessor",
    "MediaProcessorRegistry",
    "create_default_media_registry",
]
