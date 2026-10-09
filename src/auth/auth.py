from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    redirect,
    render_template_string,
    request,
    url_for,
)
from flask_login import (
    LoginManager,
    current_user,
    login_user,
    logout_user,
)

from src.db.repository import (
    create_system_log,
    get_admin_by_id,
    get_admin_by_username,
    update_admin_password_hash,
    update_last_login,
)
from src.security.password import verify_password


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


login_manager = LoginManager()


LOGIN_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <title>FlowCool 관리자 로그인</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;

            display: flex;
            align-items: center;
            justify-content: center;

            background: #f3f6fa;

            font-family:
                "Malgun Gothic",
                Arial,
                sans-serif;

            color: #17324d;
        }

        .login-card {
            width: min(420px, calc(100% - 32px));

            padding: 40px;

            background: white;

            border: 1px solid #dce4ec;
            border-radius: 16px;

            box-shadow:
                0 18px 50px
                rgba(23, 50, 77, 0.09);
        }

        .brand {
            margin-bottom: 10px;

            color: #2563eb;

            font-size: 13px;
            font-weight: 700;

            letter-spacing: 0.08em;
        }

        h1 {
            margin: 0 0 8px;
            font-size: 26px;
        }

        .subtitle {
            margin: 0 0 30px;

            color: #687686;
            font-size: 14px;
        }

        label {
            display: block;

            margin: 18px 0 7px;

            font-size: 13px;
            font-weight: 700;
        }

        input {
            width: 100%;
            height: 46px;

            padding: 0 13px;

            border: 1px solid #ccd5df;
            border-radius: 8px;

            font-size: 15px;
        }

        input:focus {
            outline: none;
            border-color: #2563eb;
        }

        button {
            width: 100%;
            height: 48px;

            margin-top: 26px;

            border: 0;
            border-radius: 8px;

            background: #17324d;
            color: white;

            font-size: 15px;
            font-weight: 700;

            cursor: pointer;
        }

        .error {
            margin-top: 18px;

            padding: 12px;

            border-radius: 7px;

            background: #fff1f2;
            color: #b42318;

            font-size: 13px;
        }
    </style>
</head>

<body>

    <div class="login-card">

        <div class="brand">
            FLOWCOOL SYSTEM
        </div>

        <h1>
            관리자 로그인
        </h1>

        <p class="subtitle">
            설비 상태 감시 시스템에 접속합니다.
        </p>

        <form method="post">

            <label for="username">
                관리자 ID
            </label>

            <input
                id="username"
                name="username"
                autocomplete="username"
                required
            >

            <label for="password">
                비밀번호
            </label>

            <input
                id="password"
                name="password"
                type="password"
                autocomplete="current-password"
                required
            >

            <button type="submit">
                로그인
            </button>

        </form>

        {% if error %}
            <div class="error">
                {{ error }}
            </div>
        {% endif %}

    </div>

</body>
</html>
"""


def get_client_ip() -> str | None:
    return request.remote_addr


def init_auth(server):

    secret_key = os.getenv(
        "FLOWCOOL_SECRET_KEY"
    )

    if not secret_key:
        raise RuntimeError(
            "FLOWCOOL_SECRET_KEY가 .env에 설정되지 않았습니다."
        )

    server.secret_key = secret_key

    server.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=(
            os.getenv(
                "FLOWCOOL_COOKIE_SECURE",
                "0",
            )
            == "1"
        ),
    )

    login_manager.init_app(server)

    login_manager.login_view = "login"
    login_manager.session_protection = "strong"

    @server.route("/")
    def home():

        if current_user.is_authenticated:
                return redirect("/dashboard/")

        return redirect("/login")

    # ========================================================
    # 사용자 로딩
    # ========================================================

    @login_manager.user_loader
    def load_user(user_id):

        try:
            return get_admin_by_id(
                int(user_id)
            )

        except (ValueError, TypeError):
            return None

    # ========================================================
    # 로그인
    # ========================================================

    @server.route(
        "/login",
        methods=["GET", "POST"],
    )
    def login():

        if current_user.is_authenticated:
            return redirect("/dashboard/")

        error = None

        if request.method == "POST":

            username = (
                request.form
                .get("username", "")
                .strip()
            )

            password = (
                request.form
                .get("password", "")
            )

            admin = get_admin_by_username(
                username
            )

            valid = False
            new_hash = None

            if (
                admin is not None
                and admin.is_active
            ):

                valid, new_hash = (
                    verify_password(
                        admin.password_hash,
                        password,
                    )
                )

            # -----------------------------
            # 로그인 성공
            # -----------------------------

            if valid:

                if new_hash:

                    update_admin_password_hash(
                        admin.id,
                        new_hash,
                    )

                login_user(
                    admin,
                    remember=False,
                )

                update_last_login(
                    admin.id
                )

                create_system_log(
                    admin_id=admin.id,
                    event_type="LOGIN_SUCCESS",
                    event_level="INFO",
                    message="관리자 로그인 성공",
                    client_ip=get_client_ip(),
                )

                next_url = request.args.get(
                    "next"
                )

                # 외부 URL Redirect 방지
                if (
                    not next_url
                    or not next_url.startswith("/")
                    or next_url.startswith("//")
                ):
                    next_url = "/dashboard/"

                return redirect(
                    next_url
                )

            # -----------------------------
            # 로그인 실패
            # -----------------------------

            create_system_log(
                event_type="LOGIN_FAILED",
                event_level="WARNING",
                message="관리자 로그인 실패",
                details={
                    "reason":
                    "invalid_credentials"
                },
                client_ip=get_client_ip(),
            )

            # ID 존재 여부 노출 방지
            error = (
                "아이디 또는 비밀번호가 "
                "올바르지 않습니다."
            )

        return render_template_string(
            LOGIN_TEMPLATE,
            error=error,
        )

    # ========================================================
    # 로그아웃
    # ========================================================

    @server.route("/logout")
    def logout():

        if current_user.is_authenticated:

            admin_id = int(
                current_user.id
            )

            create_system_log(
                admin_id=admin_id,
                event_type="LOGOUT",
                event_level="INFO",
                message="관리자 로그아웃",
                client_ip=get_client_ip(),
            )

            logout_user()

        return redirect("/login")

    # ========================================================
    # Dash 접근 보호
    # ========================================================

    @server.before_request
    def protect_dashboard():

        path = request.path

    # 인증 없이 접근 가능한 경로
        if (
            path == "/login"
            or path == "/favicon.ico"
            or path.startswith("/static/")
        ):
            return None

    # 로그인 상태면 모든 경로 허용
        if current_user.is_authenticated:
            return None

    # Dash 관련 경로는 로그인으로 이동
        if path.startswith("/dashboard/"):
            return redirect(
                url_for(
                    "login",
                    next=path,
                )
            )

    # 루트도 로그인으로 이동
        if path == "/":
            return redirect("/login")

        return None