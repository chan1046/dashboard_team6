from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from src.db.database import session_scope
from src.db.models import Admin, SystemLog


def get_admin_by_id(admin_id: int) -> Admin | None:
    with session_scope() as session:
        return session.get(Admin, admin_id)


def get_admin_by_username(username: str) -> Admin | None:
    with session_scope() as session:
        stmt = select(Admin).where(
            Admin.username == username
        )

        return session.scalar(stmt)


def create_admin(
    *,
    username: str,
    password_hash: str,
    display_name: str,
    email: str | None = None,
    phone: str | None = None,
    role: str = "admin",
) -> Admin:

    with session_scope() as session:
        existing = session.scalar(
            select(Admin).where(
                Admin.username == username
            )
        )

        if existing:
            raise ValueError(
                f"이미 존재하는 관리자 ID입니다: {username}"
            )

        admin = Admin(
            username=username,
            password_hash=password_hash,
            display_name=display_name,
            email=email,
            phone=phone,
            role=role,
        )

        session.add(admin)
        session.flush()

        return admin


def update_admin_password_hash(
    admin_id: int,
    password_hash: str,
) -> None:

    with session_scope() as session:
        admin = session.get(Admin, admin_id)

        if admin:
            admin.password_hash = password_hash


def update_last_login(admin_id: int) -> None:
    with session_scope() as session:
        admin = session.get(Admin, admin_id)

        if admin:
            admin.last_login_at = datetime.now(timezone.utc)


def create_system_log(
    *,
    event_type: str,
    message: str,
    admin_id: int | None = None,
    event_level: str = "INFO",
    tool_id: str | None = None,
    details: dict | None = None,
    client_ip: str | None = None,
) -> None:

    with session_scope() as session:

        log = SystemLog(
            admin_id=admin_id,
            event_type=event_type,
            event_level=event_level,
            tool_id=tool_id,
            message=message,
            details=details,
            client_ip=client_ip,
        )

        session.add(log)


def get_system_logs(
    *,
    limit: int = 100,
) -> list[SystemLog]:

    with session_scope() as session:

        stmt = (
            select(SystemLog)
            .order_by(SystemLog.created_at.desc())
            .limit(limit)
        )

        return list(session.scalars(stmt))