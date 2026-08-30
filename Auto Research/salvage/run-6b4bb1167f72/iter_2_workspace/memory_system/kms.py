"""External Key Management Service (KMS) abstraction.

The Architect's schema design (work order item 2) requires that AES-256-GCM
encryption keys for memory content live OUTSIDE the SQLite store, in an
external KMS.  The store therefore persists only an opaque ``key_id`` per
ciphertext blob; the key material itself is fetched from (and destroyed in) the
KMS at runtime.  A deliberately destroyed key makes the ciphertext permanently
unrecoverable *without rewriting the row*, which is the cryptographic-shredding
invariant the active-forgetting story depends on.

This module ships a deterministic, in-process ``MockKMS`` for tests and
single-process runs.  Production would swap it for a real envelope-encryption
KMS (AWS KMS / GCP KMS / a hardware HSM) behind the same three-method
interface, so the store code never changes when the KMS does.
"""

from __future__ import annotations

import os
from typing import Protocol


class KMS(Protocol):
    """The three operations the store needs from a key-management service.

    Keys are 256-bit (AES-256-GCM); ``destroy`` is *destructive* per the
    active-forgetting contract -- after it returns, ``get_key`` for that id is
    ``None`` and the ciphertext is unrecoverable.
    """

    def create_key(self, key_id: str) -> bytes:
        """Generate a fresh AES-256 key and register it under ``key_id``."""

    def get_key(self, key_id: str) -> bytes | None:
        """Return the key material, or ``None`` if unknown/destroyed."""

    def destroy_key(self, key_id: str) -> bool:
        """Destroy the key in the KMS.  Returns True if material was removed."""


class MockKMS(KMS):
    """Reference implementation holding keys only in process memory.

    This satisfies the "per-item key stored in external KMS (mocked for
    testing)" requirement with no third-party server.  It is deliberately NOT
    serialized to the store's SQLite file: if the process dies, the key is
    gone -- which is acceptable for a mock, and exercises the same
    "key absent -> ciphertext unreadable" failure mode a real KMS would.
    """

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}

    def create_key(self, key_id: str) -> bytes:
        key = os.urandom(32)  # AES-256 key material
        self._keys[key_id] = key
        return key

    def get_key(self, key_id: str) -> bytes | None:
        return self._keys.get(key_id)

    def destroy_key(self, key_id: str) -> bool:
        return self._keys.pop(key_id, None) is not None