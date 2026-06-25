"""Password encryption utilities.

Uses AES-GCM with a PBKDF2-derived key so the client can encrypt the
password before transmitting it over the network.  The shared pepper is
never sent in the clear — only the salt travels alongside the ciphertext.
"""

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

_PEPPER = os.getenv("PASSWORD_ENCRYPTION_SECRET", "copiioai-default-pepper-key-2024")


def decrypt_password(encrypted_b64: str) -> str:
    """Decrypt a password encrypted by the frontend's encryptPassword()."""
    data = base64.b64decode(encrypted_b64)
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
