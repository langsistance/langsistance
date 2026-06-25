"""Password encryption utilities.

The client encrypts the password before transmitting it, so the raw
password never appears in the Network tab request body.

Two schemes are supported, distinguished by a prefix:

  "a:<base64>" — AES-GCM with a PBKDF2-derived key (secure context)
  "x:<base64>" — XOR cipher (fallback when crypto.subtle is unavailable)
"""

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

_PEPPER = os.getenv("PASSWORD_ENCRYPTION_SECRET", "copiioai-default-pepper-key-2024")


# ── djb2 hash (matches frontend hashString) ─────────────────────────────────


def _hash_string(s: str) -> int:
    h = 5381
    for ch in s:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return h


# ── AES-GCM (prefix "a:") ──────────────────────────────────────────────────


def _decrypt_aes(payload_b64: str) -> str:
    data = base64.b64decode(payload_b64)
    salt = data[:16]
    iv = data[16:28]
    ciphertext = data[28:]

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        salt=salt,
        length=32,
        iterations=100_000,
    )
    key = kdf.derive(_PEPPER.encode())

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(iv, ciphertext, None)
    return plaintext.decode("utf-8")


# ── XOR fallback (prefix "x:") ─────────────────────────────────────────────


def _decrypt_xor(payload_b64: str) -> str:
    data = base64.b64decode(payload_b64)
    salt = data[:16]
    cipher = data[16:]

    # same seed derivation as the frontend
    salt_b64 = base64.b64encode(salt).decode()
    seed = _hash_string(_PEPPER + ":" + salt_b64)

    def _next_byte() -> int:
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return ((seed >> 7) ^ (seed & 0xFF)) & 0xFF

    result = bytearray(len(cipher))
    for i, b in enumerate(cipher):
        result[i] = b ^ _next_byte()
    return result.decode("utf-8")


# ── public API ─────────────────────────────────────────────────────────────


def decrypt_password(encrypted: str) -> str:
    """Decrypt a password produced by the frontend's encryptPassword()."""
    if encrypted.startswith("a:"):
        return _decrypt_aes(encrypted[2:])
    if encrypted.startswith("x:"):
        return _decrypt_xor(encrypted[2:])
    raise ValueError(
        "Unsupported password encryption format — expected 'a:' or 'x:' prefix"
    )
