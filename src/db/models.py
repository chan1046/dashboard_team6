from __future__ import annotations

from datetime import datetime, timezone

from flask_login import UserMixin
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from src.db.database import Base
from src.security.encryption import EncryptedText


def utc_now():
    return datetime.now(timezone.utc)


class Admin(UserMixin, Base):
    __tablename__ = "admins"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # 로그인용 ID
    # 조회가 필요하므로 암호화하지 않습니다.
    # 개인 이름/학번/전화번호를 login_id로 사용하지 않는 것을 권장합니다.
    username: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
    )

    # Argon2id Hash
    password_hash: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    # 개인정보
    # DB에는 AES-256-GCM 암호문으로 저장
    display_name: Mapped[str] = mapped_column(
        EncryptedText(),
        nullable=False,
    )

    email: Mapped[str | None] = mapped_column(
        EncryptedText(),
        nullable=True,
    )

    phone: Mapped[str | None] = mapped_column(
        EncryptedText(),
        nullable=True,
    )

    role: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="admin",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    logs: Mapped[list["SystemLog"]] = relationship(
        back_populates="admin",
    )


class SystemLog(Base):
    __tablename__ = "system_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # 로그인 실패의 경우 관리자 ID를 알 수 없을 수 있음
    admin_id: Mapped[int | None] = mapped_column(
        ForeignKey("admins.id"),
        nullable=True,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    event_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="INFO",
    )

    # 관련 설비가 있을 때
    tool_id: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
        index=True,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # 구조화된 추가 정보
    # 예:
    # {
    #   "risk_score": 0.948,
    #   "status": "위험"
    # }
    details: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # IP를 저장할 경우 개인정보 보호 차원에서 AES 암호화
    client_ip: Mapped[str | None] = mapped_column(
        EncryptedText(),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    admin: Mapped[Admin | None] = relationship(
        back_populates="logs",
    )