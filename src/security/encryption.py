from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import dotenv_values
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

AAD = b"flowcool-personal-data-v1"


def _load_key() -> bytes:
    env = dotenv_values(ENV_PATH)

    encoded_key = env.get("FLOWCOOL_AES_KEY")

    if not encoded_key:
        raise RuntimeError(
            "FLOWCOOL_AES_KEY 값이 .env에 설정되지 않았습니다."
        )

    try:
        key = base64.urlsafe_b64decode(encoded_key)

    except Exception as exc:
        raise RuntimeError(
            "FLOWCOOL_AES_KEY 형식이 올바르지 않습니다."
        ) from exc

    if len(key) != 32:
        raise RuntimeError(
            f"AES-256 Key는 정확히 32 Byte여야 합니다. "
            f"현재: {len(key)} Byte"
        )

    return key

def encrypt_text(value: str) -> str:
    if value is None:
        return value

    key = _load_key()

    aes = AESGCM(key)

    # AES-GCM 권장 nonce 크기 = 12 bytes
    nonce = os.urandom(12)

    encrypted = aes.encrypt(
        nonce,
        value.encode("utf-8"),
        AAD,
    )

    # version(1 byte) + nonce + ciphertext + tag
    payload = b"\x01" + nonce + encrypted

    return base64.urlsafe_b64encode(payload).decode("ascii")


def decrypt_text(value: str) -> str:
    if value is None:
        return value

    payload = base64.urlsafe_b64decode(value)

    version = payload[0]

    if version != 1:
        raise ValueError(
            f"지원하지 않는 암호화 데이터 버전입니다: {version}"
        )

    nonce = payload[1:13]
    encrypted = payload[13:]

    aes = AESGCM(_load_key())

    decrypted = aes.decrypt(
        nonce,
        encrypted,
        AAD,
    )

    return decrypted.decode("utf-8")


class EncryptedText(TypeDecorator):
    """
    SQLAlchemy ORM에서 문자열을 저장할 때 자동 암호화하고,
    조회할 때 자동 복호화합니다.

    검색/Unique Index가 필요한 컬럼에는 사용하지 않습니다.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None

        return encrypt_text(str(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None

        return decrypt_text(value)