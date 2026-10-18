"""按完整输入与验证协议缓存已完成的滚动验证段。"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
import uuid
import zlib
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("quant.validation_cache")
MAGIC = b"QVALIDATION1\n"


def build_cache_key(identity: dict[str, Any]) -> str:
    """将可复现的模型、数据和切分身份压缩为稳定的 SHA-256 键。"""
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ValidationCache:
    """原子保存并校验单段预测；忽略临时文件与不完整记录。"""

    def __init__(self, root: Path, max_bytes: int, ttl_days: int) -> None:
        self.root = Path(root)
        self.max_bytes = max(1, int(max_bytes))
        self.ttl_seconds = max(1, int(ttl_days)) * 24 * 60 * 60

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.vfc"

    def load(self, key: str, expected_count: int) -> dict[str, Any] | None:
        """仅返回校验通过且预测长度完全匹配的缓存记录。"""
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            stat = path.stat()
            if stat.st_size > self.max_bytes:
                raise ValueError("缓存文件超过容量上限")
            if time.time() - stat.st_mtime > self.ttl_seconds:
                path.unlink(missing_ok=True)
                return None
            contents = path.read_bytes()
            if not contents.startswith(MAGIC):
                raise ValueError("缓存文件头无效")
            checksum_start = len(MAGIC)
            checksum_end = contents.find(b"\n", checksum_start)
            if checksum_end < 0:
                raise ValueError("缓存校验头不完整")
            expected_checksum = contents[checksum_start:checksum_end].decode("ascii")
            body = gzip.decompress(contents[checksum_end + 1:])
            if hashlib.sha256(body).hexdigest() != expected_checksum:
                raise ValueError("缓存完整性校验失败")
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("缓存载荷不是对象")
            if payload.get("key") != key:
                raise ValueError("缓存键不匹配")
            predictions = np.asarray(payload.get("predictions"), dtype=float)
            if predictions.ndim != 1 or len(predictions) != int(expected_count):
                raise ValueError("缓存预测数量不匹配")
            if not np.isfinite(predictions).all():
                raise ValueError("缓存预测含非有限值")
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError("缓存元数据无效")
            os.utime(path, None)
            return {"predictions": predictions, "metadata": metadata}
        except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, zlib.error) as exc:
            LOGGER.warning("滚动验证缓存失效，重新计算 key=%s reason=%s", key[:12], type(exc).__name__)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("无权删除失效滚动验证缓存 key=%s", key[:12])
            return None

    def delete(self, key: str) -> None:
        """移除无效键对应的完整文件。"""
        try:
            self._path(key).unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("无权删除失效滚动验证缓存 key=%s", key[:12])

    def save(self, key: str, predictions: np.ndarray, metadata: dict[str, Any]) -> bool:
        """校验预测后原子写入；写入失败不影响本次计算结果。"""
        values = np.asarray(predictions, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            return False
        try:
            payload = json.dumps(
                {"key": key, "metadata": metadata, "predictions": values.tolist()},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
            compressed = gzip.compress(payload)
            contents = MAGIC + hashlib.sha256(payload).hexdigest().encode("ascii") + b"\n" + compressed
            if len(contents) > self.max_bytes:
                LOGGER.warning("滚动验证段大于缓存容量上限，跳过写入 key=%s bytes=%d", key[:12], len(contents))
                return False
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._path(key)
            temporary = self.root / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as stream:
                    stream.write(contents)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            self.cleanup()
            return True
        except (OSError, TypeError, ValueError) as exc:
            LOGGER.warning("滚动验证缓存写入失败，继续使用本次计算结果 key=%s reason=%s", key[:12], type(exc).__name__)
            return False

    def cleanup(self) -> None:
        """按过期时间和最近使用顺序限制缓存空间。"""
        if not self.root.is_dir():
            return
        now = time.time()
        cache_files = []
        for path in self.root.glob("*.vfc"):
            try:
                stat = path.stat()
                if now - stat.st_mtime > self.ttl_seconds:
                    path.unlink(missing_ok=True)
                else:
                    cache_files.append((stat.st_mtime, stat.st_size, path))
            except OSError:
                continue
        temporary_cutoff = now - self.ttl_seconds
        for path in self.root.glob(".*.tmp"):
            try:
                if path.stat().st_mtime < temporary_cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
        total = sum(size for _, size, path in cache_files if path.exists())
        for _, size, path in sorted(cache_files):
            if total <= self.max_bytes:
                break
            try:
                path.unlink(missing_ok=True)
                total -= size
            except OSError:
                continue
