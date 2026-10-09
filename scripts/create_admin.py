from getpass import getpass

from src.db.database import init_db
from src.db.repository import create_admin
from src.security.password import hash_password


def main():

    init_db()

    print("=== FlowCool 관리자 생성 ===")

    username = input(
        "관리자 로그인 ID: "
    ).strip()

    display_name = input(
        "관리자 이름: "
    ).strip()

    email = input(
        "이메일 (선택): "
    ).strip() or None

    password = getpass(
        "비밀번호: "
    )

    password_confirm = getpass(
        "비밀번호 확인: "
    )

    if password != password_confirm:
        raise ValueError(
            "비밀번호가 일치하지 않습니다."
        )

    admin = create_admin(
        username=username,
        password_hash=hash_password(password),
        display_name=display_name,
        email=email,
    )

    print(
        f"관리자가 생성되었습니다. ID={admin.id}"
    )


if __name__ == "__main__":
    main()