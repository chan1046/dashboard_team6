from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)


_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """
    평문 비밀번호를 Argon2id Hash로 변환합니다.
    """

    if not password:
        raise ValueError("비밀번호가 비어 있습니다.")

    return _password_hasher.hash(password)


def verify_password(
    password_hash: str,
    plain_password: str,
) -> tuple[bool, str | None]:
    """
    반환:
        (검증 성공 여부, 새 Hash 또는 None)

    Argon2 설정이 변경되어 rehash가 필요하면
    새로운 Hash도 함께 반환합니다.
    """

    try:
        valid = _password_hasher.verify(
            password_hash,
            plain_password,
        )

    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
    ):
        return False, None

    if not valid:
        return False, None

    new_hash = None

    if _password_hasher.check_needs_rehash(password_hash):
        new_hash = _password_hasher.hash(plain_password)

    return True, new_hash