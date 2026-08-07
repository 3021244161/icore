"""
Tests for icore.media - Multimodal file abstractions and processors.

Covers:
    - MediaType / MediaSource enumerations
    - MediaFile:
        * bytes / base64 / local / URL source resolution
        * extension / mime_type inference
        * release() and async context manager
        * validation errors on wrong source types
    - BaseMediaProcessor abstractness
    - MediaProcessorRegistry (register / get / has / list_types)
    - create_default_media_registry()
    - ImageProcessor / VideoProcessor / AudioProcessor lazy imports
        (these tests assert the ImportError path when backends absent;
         if backends ARE installed, we exercise the actual code path)
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from icore.exceptions import MediaUnsupportedError, ValidationError
from icore.media import (
    BaseMediaProcessor,
    ImageProcessor,
    AudioProcessor,
    VideoProcessor,
    MediaFile,
    MediaProcessorRegistry,
    MediaSource,
    MediaType,
    create_default_media_registry,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TestEnumerations:
    def test_media_type_values(self):
        assert MediaType.IMAGE == "image"
        assert MediaType.VIDEO == "video"
        assert MediaType.AUDIO == "audio"
        assert MediaType.PDF == "pdf"

    def test_media_source_values(self):
        assert MediaSource.LOCAL == "local"
        assert MediaSource.URL == "url"
        assert MediaSource.BASE64 == "base64"
        assert MediaSource.BYTES == "bytes"


# ---------------------------------------------------------------------------
# MediaFile - source resolution
# ---------------------------------------------------------------------------

class TestMediaFileBytes:
    async def test_to_bytes_from_bytes_source(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=b"raw-bytes",
        )
        assert await mf.to_bytes() == b"raw-bytes"

    async def test_to_bytes_caches(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=b"data",
        )
        first = await mf.to_bytes()
        second = await mf.to_bytes()
        assert first is second  # cached

    async def test_to_bytes_rejects_non_bytes(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source="not-bytes",  # type: ignore[arg-type]
        )
        with pytest.raises(ValidationError):
            await mf.to_bytes()


class TestMediaFileBase64:
    async def test_to_bytes_from_base64(self):
        original = b"hello world"
        encoded = base64.b64encode(original).decode("ascii")
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BASE64,
            source=encoded,
        )
        assert await mf.to_bytes() == original

    async def test_to_base64_round_trip(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=b"round-trip",
        )
        b64 = await mf.to_base64()
        assert base64.b64decode(b64) == b"round-trip"

    async def test_to_base64_caches(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=b"x",
        )
        first = await mf.to_base64()
        second = await mf.to_base64()
        assert first is second

    async def test_to_bytes_rejects_non_str_base64(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BASE64,
            source=b"not-str",  # type: ignore[arg-type]
        )
        with pytest.raises(ValidationError):
            await mf.to_bytes()


class TestMediaFileLocal:
    async def test_to_bytes_from_local(self, tmp_path: Path):
        path = tmp_path / "test.txt"
        path.write_bytes(b"local-data")
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.LOCAL,
            source=str(path),
        )
        assert await mf.to_bytes() == b"local-data"

    async def test_missing_local_file_raises(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.LOCAL,
            source="/nonexistent/path/file.txt",
        )
        with pytest.raises(ValidationError):
            await mf.to_bytes()


# ---------------------------------------------------------------------------
# MediaFile - properties
# ---------------------------------------------------------------------------

class TestMediaFileProperties:
    def test_extension_from_local_path(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.LOCAL,
            source="/tmp/foo.JPEG",
        )
        assert mf.extension == ".jpeg"

    def test_extension_from_metadata_override(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=b"",
            metadata={"extension": "png"},
        )
        assert mf.extension == ".png"

    def test_extension_default_by_type(self):
        mf = MediaFile(
            media_type=MediaType.AUDIO,
            source_type=MediaSource.BYTES,
            source=b"",
        )
        # Default audio extension from _EXT_BY_TYPE.
        assert mf.extension == ".mp3"

    def test_mime_type_from_path(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.LOCAL,
            source="/tmp/photo.png",
        )
        assert mf.mime_type == "image/png"

    def test_mime_type_metadata_override(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=b"",
            metadata={"mime_type": "image/webp"},
        )
        assert mf.mime_type == "image/webp"

    def test_mime_type_default_by_type(self):
        mf = MediaFile(
            media_type=MediaType.VIDEO,
            source_type=MediaSource.BYTES,
            source=b"",
        )
        assert mf.mime_type == "video/mp4"


# ---------------------------------------------------------------------------
# MediaFile - context manager
# ---------------------------------------------------------------------------

class TestMediaFileContextManager:
    async def test_release_clears_caches(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=b"x",
        )
        await mf.to_bytes()
        assert mf._raw_bytes is not None
        mf.release()
        assert mf._raw_bytes is None
        assert mf._base64 is None

    async def test_async_context_manager_releases_on_exit(self):
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=b"y",
        )
        async with mf:
            await mf.to_bytes()
            assert mf._raw_bytes is not None
        # After exit, caches released.
        assert mf._raw_bytes is None


# ---------------------------------------------------------------------------
# Abstract processor
# ---------------------------------------------------------------------------

class TestBaseMediaProcessor:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseMediaProcessor()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestMediaProcessorRegistry:
    def test_register_and_get(self):
        reg = MediaProcessorRegistry()
        proc = ImageProcessor()
        reg.register(MediaType.IMAGE, proc)
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=b"",
        )
        assert reg.get(mf) is proc

    def test_get_unregistered_raises(self):
        reg = MediaProcessorRegistry()
        mf = MediaFile(
            media_type=MediaType.AUDIO,
            source_type=MediaSource.BYTES,
            source=b"",
        )
        with pytest.raises(MediaUnsupportedError):
            reg.get(mf)

    def test_has(self):
        reg = MediaProcessorRegistry()
        reg.register(MediaType.IMAGE, ImageProcessor())
        assert reg.has(MediaType.IMAGE) is True
        assert reg.has(MediaType.AUDIO) is False

    def test_list_types(self):
        reg = MediaProcessorRegistry()
        reg.register(MediaType.IMAGE, ImageProcessor())
        reg.register(MediaType.AUDIO, AudioProcessor())
        types = reg.list_types()
        assert MediaType.IMAGE in types
        assert MediaType.AUDIO in types

    def test_create_default_registry(self):
        reg = create_default_media_registry()
        assert reg.has(MediaType.IMAGE)
        assert reg.has(MediaType.VIDEO)
        assert reg.has(MediaType.AUDIO)


# ---------------------------------------------------------------------------
# ImageProcessor - lazy import path
# ---------------------------------------------------------------------------

class TestImageProcessorLazy:
    def test_module_importable_without_pillow(self):
        # The fact that this test file imports cleanly proves it.
        from icore.media import ImageProcessor  # noqa: F401
        assert ImageProcessor is not None

    async def test_extract_metadata_missing_pillow(self):
        try:
            from PIL import Image  # type: ignore  # noqa: F401
            pytest.skip("Pillow installed; cannot test ImportError path")
        except ImportError:
            proc = ImageProcessor()
            mf = MediaFile(
                media_type=MediaType.IMAGE,
                source_type=MediaSource.BYTES,
                source=b"x",
            )
            with pytest.raises(ImportError):
                await proc.extract_metadata(mf)

    async def test_extract_metadata_with_pillow(self, tmp_path: Path):
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            pytest.skip("Pillow not installed")
        # Create a real PNG.
        path = tmp_path / "test.png"
        img = Image.new("RGB", (32, 16), color="red")
        img.save(path, format="PNG")
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.LOCAL,
            source=str(path),
        )
        proc = ImageProcessor()
        meta = await proc.extract_metadata(mf)
        assert meta["width"] == 32
        assert meta["height"] == 16
        assert meta["format"] == "PNG"

    async def test_transcode_with_pillow(self):
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            pytest.skip("Pillow not installed")
        import io
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color="blue").save(buf, format="PNG")
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=buf.getvalue(),
        )
        proc = ImageProcessor()
        out = await proc.transcode(mf, "jpeg")
        new_bytes = await out.to_bytes()
        assert len(new_bytes) > 0
        # Should be readable as JPEG
        Image.open(io.BytesIO(new_bytes)).verify()

    async def test_generate_thumbnail_with_pillow(self):
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            pytest.skip("Pillow not installed")
        import io
        buf = io.BytesIO()
        Image.new("RGB", (800, 600), color="green").save(buf, format="PNG")
        mf = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=buf.getvalue(),
        )
        proc = ImageProcessor()
        thumb = await proc.generate_thumbnail(mf, size=(64, 64))
        thumb_bytes = await thumb.to_bytes()
        # Verify thumbnail dimensions
        thumb_img = Image.open(io.BytesIO(thumb_bytes))
        assert max(thumb_img.size) <= 64


# ---------------------------------------------------------------------------
# VideoProcessor - lazy import path
# ---------------------------------------------------------------------------

class TestVideoProcessorLazy:
    def test_module_importable(self):
        from icore.media import VideoProcessor  # noqa: F401
        assert VideoProcessor is not None

    async def test_extract_metadata_bytes_source(self):
        # No ffprobe needed for bytes source — returns size only.
        proc = VideoProcessor()
        mf = MediaFile(
            media_type=MediaType.VIDEO,
            source_type=MediaSource.BYTES,
            source=b"fake-video-data",
        )
        meta = await proc.extract_metadata(mf)
        assert meta["size_bytes"] == len(b"fake-video-data")

    async def test_transcode_requires_local(self):
        proc = VideoProcessor()
        mf = MediaFile(
            media_type=MediaType.VIDEO,
            source_type=MediaSource.BYTES,
            source=b"x",
        )
        with pytest.raises(ValidationError):
            await proc.transcode(mf, "mp4")

    async def test_generate_thumbnail_requires_local(self):
        proc = VideoProcessor()
        mf = MediaFile(
            media_type=MediaType.VIDEO,
            source_type=MediaSource.BYTES,
            source=b"x",
        )
        with pytest.raises(ValidationError):
            await proc.generate_thumbnail(mf)

    async def test_extract_keyframes_requires_local(self):
        proc = VideoProcessor()
        mf = MediaFile(
            media_type=MediaType.VIDEO,
            source_type=MediaSource.BYTES,
            source=b"x",
        )
        with pytest.raises(ValidationError):
            await proc.extract_keyframes(mf)

    async def test_extract_audio_track_requires_local(self):
        proc = VideoProcessor()
        mf = MediaFile(
            media_type=MediaType.VIDEO,
            source_type=MediaSource.BYTES,
            source=b"x",
        )
        with pytest.raises(ValidationError):
            await proc.extract_audio_track(mf)


# ---------------------------------------------------------------------------
# AudioProcessor - lazy import path
# ---------------------------------------------------------------------------

class TestAudioProcessorLazy:
    def test_module_importable(self):
        from icore.media import AudioProcessor  # noqa: F401
        assert AudioProcessor is not None

    async def test_extract_metadata_missing_pydub(self):
        try:
            from pydub import AudioSegment  # type: ignore  # noqa: F401
            pytest.skip("pydub installed; cannot test ImportError path")
        except ImportError:
            proc = AudioProcessor()
            mf = MediaFile(
                media_type=MediaType.AUDIO,
                source_type=MediaSource.BYTES,
                source=b"x",
            )
            with pytest.raises(ImportError):
                await proc.extract_metadata(mf)

    async def test_transcribe_missing_whisper(self):
        try:
            import whisper  # type: ignore  # noqa: F401
            pytest.skip("whisper installed; cannot test ImportError path")
        except ImportError:
            proc = AudioProcessor()
            mf = MediaFile(
                media_type=MediaType.AUDIO,
                source_type=MediaSource.LOCAL,
                source="/tmp/foo.mp3",
            )
            with pytest.raises(ImportError):
                await proc.transcribe(mf)

    async def test_generate_thumbnail_raises(self):
        # Audio thumbnail not implemented.
        proc = AudioProcessor()
        mf = MediaFile(
            media_type=MediaType.AUDIO,
            source_type=MediaSource.BYTES,
            source=b"x",
        )
        with pytest.raises(MediaUnsupportedError):
            await proc.generate_thumbnail(mf)
