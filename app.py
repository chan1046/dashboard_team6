from datetime import date

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    ctx,
    no_update,
    ALL,
)


from src.dashboard_inference import TOOLS, inspect_tools
from src.dashboard_replay import (
    REFERENCE_TIME,
    filtered_faults,
    replay_results,
    sensor_series,
)

from flask_login import current_user

from src.auth.auth import init_auth
from src.db.database import database_is_connected, init_db
from src.db.repository import create_system_log
from flask import Flask
# ============================================================
# APP
# ============================================================
server = Flask(__name__)

init_db()
init_auth(server)

app = Dash(
    __name__,
    server=server,
    routes_pathname_prefix="/dashboard/",
    requests_pathname_prefix="/dashboard/",
)
# ============================================================
# 기본 데이터
# 저장 모델과 실제 검증 데이터 재생
# ============================================================

UNITS = [{"id": tool} for tool in TOOLS]

SENSORS = [
    "FLOWCOOLPRESSURE",
    "FLOWCOOLFLOWRATE",
    "IONGAUGEPRESSURE",
    "ETCHBEAMCURRENT",
    "ETCHGASCHANNEL1READBACK",
]

SENSOR_LABELS = {
    "FLOWCOOLPRESSURE": "냉각 압력",
    "FLOWCOOLFLOWRATE": "냉각 유량",
    "IONGAUGEPRESSURE": "이온 게이지 압력",
    "ETCHBEAMCURRENT": "식각 빔 전류",
    "ETCHGASCHANNEL1READBACK": "식각 가스 채널",
}

ALL_SENSORS = [
    "IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE", "ETCHSUPPRESSORCURRENT", "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE", "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE", "ROTATIONSPEED", "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION", "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER", "ACTUALSTEPDURATION",
]

SENSOR_LABELS.update({
    "ETCHBEAMVOLTAGE": "식각 빔 전압", "ETCHSUPPRESSORVOLTAGE": "억제 전압",
    "ETCHSUPPRESSORCURRENT": "억제 전류", "ETCHPBNGASREADBACK": "PB 가스 유량",
    "FIXTURETILTANGLE": "치구 기울기", "ROTATIONSPEED": "회전 속도",
    "ACTUALROTATIONANGLE": "실제 회전 각도", "FIXTURESHUTTERPOSITION": "치구 셔터 위치",
    "ETCHSOURCEUSAGE": "식각 소스 사용량", "ETCHAUXSOURCETIMER": "보조 소스 타이머 1",
    "ETCHAUX2SOURCETIMER": "보조 소스 타이머 2", "ACTUALSTEPDURATION": "실제 공정 단계 시간",
})

FAULT_LABELS = {
    "pressure_low": "냉각 압력 저하", "Low Pressure": "냉각 압력 저하",
    "pressure_high": "냉각 압력 과다", "High Pressure": "냉각 압력 과다",
    "leak": "냉각수 누수", "Leak": "냉각수 누수",
}

FAULT_RESPONSE_GUIDES = {
    "pressure_low": {
        "checks": [
            "FlowCool 압력 상태를 우선 확인하세요.",
            "FLOWCOOLFLOWRATE를 함께 확인해 헬륨 공급 및 유량 상태에 이상이 있는지 점검하세요.",
        ],
        "sensors": ["FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE"],
    },
    "pressure_high": {
        "checks": [
            "FlowCool Pump 상태를 우선 확인하세요.",
            "FLOWCOOLPRESSURE와 FLOWCOOLFLOWRATE의 변화를 함께 확인해 냉각 회로 상태를 점검하세요.",
        ],
        "sensors": ["FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE"],
    },
    "leak": {
        "checks": [
            "FlowCool 회로의 누설 가능성을 확인하세요.",
            "FLOWCOOLPRESSURE와 IONGAUGEPRESSURE 변화를 비교하고, wafer/fixture의 O-ring 및 sealing 상태와 chamber vacuum 상태를 우선 점검하세요.",
        ],
        "sensors": ["FLOWCOOLPRESSURE", "IONGAUGEPRESSURE"],
    },
}

FAULT_GUIDE_KEYS = {
    "Low Pressure": "pressure_low",
    "High Pressure": "pressure_high",
    "Leak": "leak",
}

FAULT_TYPE_COLORS = {
    "pressure_low": "#123B73",
    "pressure_high": "#1F78D1",
    "leak": "#1597C5",
}

PERIOD_OPTIONS = [
    {"label": "최근 12시간", "value": 12},
    {"label": "최근 24시간", "value": 24},
    {"label": "최근 3일", "value": 72},
    {"label": "최근 일주일", "value": 168},
    {"label": "최근 한 달", "value": 720},
    {"label": "최근 6개월", "value": 4320},
    {"label": "전체 기간", "value": 999},
]


def display_fault(value):
    return FAULT_LABELS.get(value, value or "판단 보류")


def build_recommended_actions(model2):
    """Build display-only inspection guidance from the saved Model 2 class."""
    predicted_class = (model2 or {}).get("predicted_class")
    guide_key = FAULT_GUIDE_KEYS.get(predicted_class, predicted_class)
    guide = FAULT_RESPONSE_GUIDES.get(guide_key) if (model2 or {}).get("available") else None

    if not guide:
        return html.Section([
            html.H2("추천 점검 항목"),
            html.P("예상 고장 유형과 센서 상태를 기준으로 우선 확인할 항목입니다.", className="muted"),
            html.Div([
                html.Strong("추천 점검 항목을 생성할 수 없습니다."),
                html.Span("예상 고장 유형을 먼저 확인해 주세요."),
            ], className="recommended-actions-empty"),
            html.Div(
                html.Button("조치 기록 →", id={"type": "recommended-response", "source": "guide"}, className="nav-action next"),
                className="recommended-actions-footer",
            ),
        ], className="panel recommended-actions")

    return html.Section([
        html.H2("추천 점검 항목"),
        html.P("예상 고장 유형과 센서 상태를 기준으로 우선 확인할 항목입니다.", className="muted"),
        html.Div([
            html.Div([
                html.Span("예상 고장 유형", className="recommended-actions-label"),
                html.Strong(display_fault(predicted_class), className="recommended-actions-fault"),
            ], className="recommended-actions-type"),
            html.Div([
                html.Span("우선 점검", className="recommended-actions-label"),
                html.Ul([html.Li(check) for check in guide["checks"]]),
            ], className="recommended-actions-checks"),
        ], className="recommended-actions-grid"),
        html.Div([
            html.Span("확인 센서", className="recommended-actions-label"),
            html.Div([html.Code(sensor) for sensor in guide["sensors"]], className="recommended-actions-sensors"),
        ], className="recommended-actions-sensor-group"),
        html.P(
            "※ 공개 PHM 설비 구조와 고장 메커니즘을 기반으로 한 점검 가이드이며, 실제 정비 절차는 현장 기준을 따릅니다.",
            className="recommended-actions-note",
        ),
        html.Div(
            html.Button("조치 기록 →", id={"type": "recommended-response", "source": "guide"}, className="nav-action next"),
            className="recommended-actions-footer",
        ),
    ], className="panel recommended-actions")


def relative_time(seconds):
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    if days >= 60:
        months, remaining_days = divmod(days, 30)
        return f"{months}개월 {remaining_days}일 전" if remaining_days else f"{months}개월 전"
    if days:
        return f"{days}일 {hours}시간 전" if hours else f"{days}일 전"
    if hours:
        return f"{hours}시간 {minutes}분 전" if minutes else f"{hours}시간 전"
    return f"{minutes}분 전" if minutes else "분석 기준시점"

STEP_NAMES = [
    "모니터링 설정",
    "설비 현황",
    "우선 점검 순서",
    "상세 진단",
    "조치 기록",
]


# ============================================================
# Helper Components
# ============================================================
def current_admin_id():

    if not current_user.is_authenticated:
        return None

    return int(
        current_user.id
    )

def kpi(label, note="", cls="", value="—"):
    return html.Div(
        className=f"kpi {cls}",
        children=[
            html.Div(label, className="label"),
            html.Div(value, className="metric-placeholder"),
            html.Div(note, className="note"),
        ],
    )


def placeholder(text, extra_class=""):
    return html.Div(
        text,
        className=f"placeholder {extra_class}",
    )


def page_head(number, title, subtitle, action=None, context_id=None):
    children = [
        html.Div(
            [
                *([number] if number is not None else []),
                *([html.H1(title)] if title is not None else []),
                *([html.P(subtitle, className="muted")] if subtitle is not None else []),
            ]
        )
    ]

    if context_id is not None or action is not None:
        children.append(html.Div([
            context_bar(context_id) if context_id is not None else None,
            action,
        ], className="pagehead-actions"))

    return html.Div(
        children,
        className="pagehead",
    )


def context_bar(context_id):
    period_id = {
        "context-1": "period-overview",
        "context-2": "period-priority",
        "context-3": "period-diagnosis",
    }[context_id]
    return html.Div(
        [
            html.Span(id=f"{context_id}-target"),
            html.Div([
                html.Span("조회 기간", className="context-period-label"),
                dcc.Dropdown(
                    id=period_id,
                    options=PERIOD_OPTIONS,
                    value=72,
                    clearable=False,
                    searchable=False,
                    className="context-period-dropdown",
                ),
            ], className="context-period-field"),
        ],
        id=context_id,
        className="context",
    )


# ============================================================
# Header
# ============================================================

header = html.Header(
    [
        html.Div(
            className="brand",
            children=[
                html.Span("S", className="logo"),
                html.Span(
                    [
                        "냉각 설비 관리 및 대응 시스템",
                    ]
                ),
            ],
        ),

        html.Div(
            className="header-right",
            children=[
                html.Button(
                    "운영 흐름",
                    id="flowBtn",
                    className="quiet",
                ),

        html.A(
            "로그아웃",
            href="/logout",
            className="quiet logout-link",
        ),
            ],
        ),
    ]
)

loading_overlay = html.Div(
    id="analysis-loading",
    className="analysis-loading",
    style={"display": "none"},
    children=[html.Div([
        html.H2("설비 상태를 분석하고 있습니다"),
        html.P("센서 입력과 고장 위험도 분석이 완료되면 자동으로 이동합니다."),
        html.Div(html.Span(), className="loading-simple-bar"),
    ], className="loading-card")],
)


# ============================================================
# Navigation
# ============================================================

nav = html.Nav(
    [
        html.Button(
            [
                html.B(
                    str(i + 1).zfill(2)
                ),
                name,
            ],
            id=f"nav-{i}",
        )
        for i, name in enumerate(STEP_NAMES)
    ],
    id="steps",
)

# ============================================================
# PAGE 01
# 검사 설정
# ============================================================

page_setup = html.Div(
    id="page-0",
    children=[

        page_head(
            None,
            "설비 모니터링 시작",
            "대상과 조회 범위를 선택해 현재 위험도와 점검 순서를 확인하세요.",
        ),

        html.Div(
            className="setup",
            children=[

                html.Section(
                    className="panel",
                    children=[

                        html.Div([
                            html.H2("모니터링 범위"),
                            html.P("전체 설비 현황을 보거나 한 대를 바로 상세 진단할 수 있습니다.", className="muted"),
                        ], className="panel-intro"),

                        html.Div(
                            className="field",
                            children=[
                                html.Label(
                                    "대상 설비"
                                ),

                                dcc.Dropdown(
                                    id="scope",
                                    options=[
                                        {
                                            "label": "전체설비",
                                            "value": "all",
                                        }
                                    ]
                                    + [
                                        {
                                            "label": unit["id"],
                                            "value": str(i),
                                        }
                                        for i, unit
                                        in enumerate(UNITS)
                                    ],
                                    value="all",
                                    clearable=False,
                                    searchable=False,
                                ),
                            ],
                        ),

                        html.Div(
                            className="field",
                            children=[
                                html.Label("고장 이력 조회 범위"),

                                dcc.Dropdown(
                                    id="period",
                                    options=PERIOD_OPTIONS,
                                    value=72,
                                    clearable=False,
                                    searchable=False,
                                ),

                                html.Span(
                                    "예측은 저장된 검증 구간의 최신 관측점을 사용합니다.",
                                    className="muted",
                                ),
                            ],
                        ),

                        html.Button(
                            [
                                "상태 조회",
                            ],
                            id="start-btn",
                            className="primary",
                        ),

                        dcc.Loading(html.Div(id="inspection-status", role="status")),

                        html.P(
                            "판단 보류 설비는 위험도 순위에서 제외하고 센서 입력 상태를 먼저 확인합니다.",
                            className="muted",
                        ),
                    ],
                ),

                html.Aside(
                    className="blueprint",
                    children=[
                        html.H2(
                            "현장 판단 순서",
                            style={
                                "marginTop": "0"
                            },
                        ),

                        html.Div(
                            [
                                html.B("01"),
                                html.Span("데이터 조회"),
                            ],
                            className="path",
                        ),

                        html.Div(
                            [
                                html.B("02"),
                                html.Span(
                                    "24시간 내 고장 위험 분석"
                                ),
                            ],
                            className="path",
                        ),

                        html.Div(
                            [
                                html.B("03"),
                                html.Span(
                                    "점검 우선순위 산정"
                                ),
                            ],
                            className="path",
                        ),

                        html.Div(
                            [
                                html.B("04"),
                                html.Span(
                                    "예상 고장 유형 분석"
                                ),
                            ],
                            className="path",
                        ),

                        html.Div(
                            [
                                html.B("05"),
                                html.Span(
                                    "조치 기록 및 인계"
                                ),
                            ],
                            className="path",
                        ),

                        html.Div(
                            className="setupnote",
                            children=[
                                html.P(
                                    "위험 설비부터 확인하고, 센서 근거와 과거 고장을 검토한 뒤 조치 내용을 남깁니다.",
                                    className="muted",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================
# PAGE 02
# 전체 검사 결과
# ============================================================

page_overview = html.Div(
    id="page-1",
    children=[

        page_head(
            None,
            "전체 설비 현황",
            "현재 위험 분포와 실제 고장 이력을 함께 확인합니다.",
            html.Button(
                "우선 점검 순서 →",
                id="to-priority",
                className="nav-action next",
            ),
            context_id="context-1",
        ),

        html.Div(
            className="section-title",
            children=[
                html.H2("현재 운영 상태")
            ],
        ),

        html.Div(id="overview-kpis", className="kpis"),
        html.Div(id="overview-results"),

        html.Div(
            className="section-title",
            children=[
                html.H2("고장 이력 분석")
            ],
        ),

        html.Section(
            className="panel",
            children=[
                html.H3("시간대별 고장 발생 추이"),
                dcc.Graph(id="overview-fault-trend", config={"displayModeBar": False}),
            ],
        ),

        html.Section(
            className="panel forecast overview-summary-stack",
            children=[
                html.Div([
                    html.Div([
                        html.H2("고장 유형 구성"),
                        html.P("선택 기간에 발생한 고장 유형별 비율", className="muted"),
                        dcc.Graph(id="overview-fault-types", config={"displayModeBar": False}),
                    ], className="fault-donut-panel"),
                    html.Div([
                        html.H2("기간 내 실제 고장 요약"),
                        html.Div(id="forecast-results"),
                    ], className="actual-fault-summary"),
                ], className="fault-composition-grid"),
            ],
        ),
    ],
)


# ============================================================
# PAGE 03
# 우선순위
# ============================================================

page_priority = html.Div(
    id="page-2",
    children=[

        page_head(
            None,
            "우선 점검 순서",
            "24시간 내 고장 위험이 높은 설비부터 확인합니다.",
            html.Button(
                "← 설비 현황",
                id="priority-back",
                className="nav-action prev",
            ),
            context_id="context-2",
        ),

        html.Div([
            html.Span("정렬 기준", className="priority-sort-label"),
            dcc.Dropdown(
                id="priority-sort",
                options=[
                    {"label": "위험도 높은 순", "value": "risk"},
                    {"label": "예상 고장 유형별", "value": "fault"},
                    {"label": "기간 내 고장 순", "value": "history"},
                ],
                value="risk",
                clearable=False,
                searchable=False,
                className="priority-sort-dropdown",
            ),
        ], className="priority-sortbar"),

        html.Section(
            className="panel",
            style={"padding": 0},
            children=[

                html.Div(
                    className="tablewrap",
                    children=[

                        html.Div(id="priority-results"),
                    ],
                ),


            ],
        ),
    ],
)


# ============================================================
# PAGE 04
# 상세진단
# ============================================================

page_diagnosis = html.Div(
    id="page-3",
    children=[

        page_head(
            html.Span("점검 순위 미확인", id="detail-priority", className="eyebrow"),
            html.Span(
                "설비 상세진단",
                id="detail-title",
            ),
            "고장 이력과 센서 흐름을 바탕으로 "
            "예측 결과를 검토합니다.",
            html.Button(
                "← 우선 점검 순서",
                id="detail-back",
                className="nav-action prev",
            ),
            context_id="context-3",
        ),

        html.Div(id="detail-history", className="two equal", style={"marginBottom": "24px"}),
        html.Section(id="detail-status", className="detailstatus"),
        html.Div(id="detail-errors", role="status", className="muted"),

        html.Div(
            className="section-title",
            children=[
                html.H2("과거 고장 이력")
            ],
        ),

        html.Div(
            className="diagnosis-history-stack",
            children=[

                html.Section(
                    className="panel",
                    children=[
                        html.H3("기간 내 고장 발생 시점"),

                        dcc.Graph(id="detail-fault-trend", config={"displayModeBar": False}),
                        html.Div(id="detail-fault-list", className="fault-list"),
                    ],
                ),

                html.Section(
                    className="panel",
                    children=[
                        html.H3(
                            "고장 유형별 발생"
                        ),

                        dcc.Graph(id="detail-fault-types", config={"displayModeBar": False}),
                    ],
                ),
            ],
        ),

        html.Section(
            className="panel",
            style={
                "marginTop": "24px"
            },
            children=[

                html.Div([
                    html.H2("센서 데이터 추이"),
                ], className="section-title sensor-head"),

                html.Div(
                    className="sensorbuttons",
                    children=[
                        html.Button(SENSOR_LABELS[sensor], id=f"sensor-{i}", className="small")
                        for i, sensor
                        in enumerate(SENSORS)
                    ],
                ),

                html.Details([
                    html.Summary("전체 변수 보기"),
                    dcc.Dropdown(
                        id="all-sensor-select",
                        options=[{"label": SENSOR_LABELS[name], "value": name}
                                 for name in ALL_SENSORS],
                        value=SENSORS[0], clearable=False,
                    ),
                ], className="all-sensors"),

                html.Div(id="sensor-chart", className="sensor-chart"),
            ],
        ),

        html.Div(id="recommended-actions"),
        html.Button("조치 기록 →", id="to-response", style={"display": "none"}),
    ],
)


# ============================================================
# PAGE 05
# 대응 및 피드백
# ============================================================

page_response = html.Div(
    id="page-4",
    children=[

        page_head(
            None,
            "조치 기록 및 인계",
            "점검 내용과 조치 결과를 기록하고 "
            "다음 근무자에게 전달합니다.",
            html.Button(
                "← 상세 진단",
                id="response-back",
                className="nav-action prev",
            ),
        ),

        html.Div(
            id="response-context",
            className="context",
        ),

        html.Div(
            className="two equal",
            children=[

                html.Section(
                    className="panel",
                    children=[

                        html.H2(
                            "대응 보고서 작성"
                        ),

                        html.Div(
                            className="two equal",
                            children=[

                                html.Div(
                                    className="field",
                                    children=[
                                        html.Label(
                                            "작업자"
                                        ),

                                        dcc.Input(
                                            id="worker",
                                            type="text",
                                            placeholder="이름 입력",
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="field",
                                    children=[
                                        html.Label(
                                            "보고 날짜"
                                        ),

                                        dcc.Input(
                                            id="report-date",
                                            type="date",
                                            value=str(
                                                date.today()
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        html.Div(
                            className="field",
                            children=[

                                html.Label(
                                    "진행 상태"
                                ),

                                dcc.Dropdown(
                                    id="report-status",
                                    options=[
                                        "접수",
                                        "점검 중",
                                        "조치 완료",
                                        "경과 관찰",
                                    ],
                                    value="접수",
                                    clearable=False,
                                ),
                            ],
                        ),

                        html.Div(
                            className="field",
                            children=[

                                html.Label(
                                    "이상 상황 및 고장 내용"
                                ),

                                dcc.Textarea(
                                    id="issue",
                                    placeholder=(
                                        "발생 시점과 확인한 "
                                        "현상을 기록하세요."
                                    ),
                                ),
                            ],
                        ),

                        html.Div(
                            className="field",
                            children=[

                                html.Label(
                                    "대응 내용 및 인계 사항"
                                ),

                                dcc.Textarea(
                                    id="action",
                                    placeholder=(
                                        "점검·조치 결과와 "
                                        "추가 확인할 내용을 "
                                        "기록하세요."
                                    ),
                                ),
                            ],
                        ),

                        html.P(
                            id="save-error",
                            className="error",
                        ),

                        html.Button(
                            "보고서 저장",
                            id="save-report",
                            className="primary",
                        ),
                    ],
                ),

                html.Section(
                    className="panel",
                    children=[

                        html.H2(
                            "누적 대응 기록"
                        ),

                        html.Div(
                            id="report-list"
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================
# Guide Modal
# ============================================================

guide_modal = html.Div(
    id="guide-modal",
    className="modal-overlay",
    style={"display": "none"},
    children=[

        html.Div(
            className="modal-box",
            children=[

                html.Button(
                    "×",
                    id="close-guide",
                    className="close",
                ),

                html.Span(
                    "WORKFLOW",
                    className="eyebrow",
                ),

                html.H2(
                    "하나의 검사, 두 가지 경로"
                ),

                html.Div(
                    "검사 대상·기간 선택 → 검사 진행",
                    className="flowline",
                ),

                html.Div(
                    className="two",
                    children=[

                        html.Section(
                            [
                                html.H3(
                                    "전체설비"
                                ),
                                html.P(
                                    "전체 결과"
                                ),
                                html.P("↓"),
                                html.P(
                                    "24시간 내 고장 위험도 기반 우선순위"
                                ),
                                html.P("↓"),
                                html.P(
                                    "장비 상세진단 "
                                    "→ 대응 조치 및 피드백"
                                ),
                            ]
                        ),

                        html.Section(
                            [
                                html.H3(
                                    "단일설비"
                                ),
                                html.P(
                                    "장비 상세진단"
                                ),
                                html.P("↓"),
                                html.P(
                                    "대응 조치 및 피드백"
                                ),

                                html.P(
                                    "전체 결과와 "
                                    "우선순위를 건너뜁니다.",
                                    className="muted",
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================
# Layout
# ============================================================

app.layout = html.Div(
    [

        dcc.Store(
            id="app-state",
            data={
                "page": 0,
                "scope": "all",
                "hours": 72,
                "unit": 0,
                "done": False,
                "sensor": 0,
                "sensor_variable": SENSORS[0],
            },
        ),

        dcc.Store(id="prediction-results", data={}),
        dcc.Store(id="inspection-hours", data=72),

        dcc.Store(
            id="reports-store",
            storage_type="local",
            data=[],
        ),

        header,
        nav,
        loading_overlay,

        html.Main(
            [
                page_setup,
                page_overview,
                page_priority,
                page_diagnosis,
                page_response,
            ],
            id="app",
        ),

        html.Footer(
            [
                html.Span(
                    "FlowCool 예지보전 시스템 · 6팀"
                ),
                html.Span(
                    "설비 상태 감시 대시보드"
                ),
            ]
        ),

        guide_modal,
    ]
)


# ============================================================
# 상태 변경
# ============================================================

@app.callback(
    Output(
        "app-state",
        "data",
    ),
    [
        Input(
            "prediction-results",
            "data",
        ),
        Input(
            "nav-0",
            "n_clicks",
        ),
        Input(
            "nav-1",
            "n_clicks",
        ),
        Input(
            "nav-2",
            "n_clicks",
        ),
        Input(
            "nav-3",
            "n_clicks",
        ),
        Input(
            "nav-4",
            "n_clicks",
        ),
        Input(
            "to-priority",
            "n_clicks",
        ),
        Input(
            "priority-back",
            "n_clicks",
        ),
        Input(
            {"type": "detail-btn", "tool": ALL, "view": ALL},
            "n_clicks",
        ),
        Input(
            "detail-back",
            "n_clicks",
        ),
        Input(
            "to-response",
            "n_clicks",
        ),
        Input(
            {"type": "recommended-response", "source": ALL},
            "n_clicks",
        ),
        Input(
            "response-back",
            "n_clicks",
        ),
        Input(
            "sensor-0",
            "n_clicks",
        ),
        Input(
            "sensor-1",
            "n_clicks",
        ),
        Input(
            "sensor-2",
            "n_clicks",
        ),
        Input(
            "sensor-3",
            "n_clicks",
        ),
        Input(
            "sensor-4",
            "n_clicks",
        ),
        Input("all-sensor-select", "value"),
        Input("period", "value"),
        Input("period-overview", "value"),
        Input("period-priority", "value"),
        Input("period-diagnosis", "value"),
    ],
    [
        State(
            "app-state",
            "data",
        ),
        State(
            "scope",
            "value",
        ),
    ],
    prevent_initial_call=True,
)
def update_state(
    start,
    nav0,
    nav1,
    nav2,
    nav3,
    nav4,
    to_priority,
    priority_back,
    detail_btn,
    detail_back,
    to_response,
    recommended_response,
    response_back,
    sensor0,
    sensor1,
    sensor2,
    sensor3,
    sensor4,
    all_sensor,
    period,
    overview_period,
    priority_period,
    diagnosis_period,
    state,
    scope,
):

    trigger = ctx.triggered_id

    state = dict(state)

    if trigger == "prediction-results":

        if not start:
            return no_update

        state["done"] = True

        if state["page"] == 0 or not state.get("scope"):
            state["scope"] = scope
            state["unit"] = 0 if scope == "all" else int(scope)
            state["page"] = 1 if scope == "all" else 3

    elif trigger in ("period", "period-overview", "period-priority", "period-diagnosis"):
        selected_hours = int({
            "period": period,
            "period-overview": overview_period,
            "period-priority": priority_period,
            "period-diagnosis": diagnosis_period,
        }[trigger])
        if state.get("hours") == selected_hours:
            return no_update
        state["hours"] = selected_hours

    elif isinstance(trigger, str) and trigger.startswith(
        "nav-"
    ):

        page = int(
            trigger.split("-")[1]
        )

        if page == 0:
            state["page"] = 0

        elif state["done"]:

            if (
                state["scope"] != "all"
                and page in [1, 2]
            ):
                return state

            state["page"] = page

    elif trigger == "to-priority":
        state["page"] = 2

    elif trigger == "priority-back":
        state["page"] = 1

    elif isinstance(trigger, dict) and trigger.get("type") == "detail-btn":
        if not any(detail_btn or []):
            return no_update
        state["unit"] = TOOLS.index(trigger["tool"])
        state["page"] = 3

    elif trigger == "detail-back":

        state["page"] = (
            2
            if state["scope"] == "all"
            else 0
        )

    elif trigger == "to-response":
        state["page"] = 4

    elif isinstance(trigger, dict) and trigger.get("type") == "recommended-response":
        if not any(recommended_response or []):
            return no_update
        state["page"] = 4

    elif trigger == "response-back":
        state["page"] = 3

    elif trigger in [
        "sensor-0",
        "sensor-1",
        "sensor-2",
        "sensor-3",
        "sensor-4",
    ]:

        state["sensor"] = int(
            trigger.split("-")[1]
        )
        state["sensor_variable"] = SENSORS[state["sensor"]]

    elif trigger == "all-sensor-select":
        state["sensor_variable"] = all_sensor
        state["sensor"] = SENSORS.index(all_sensor) if all_sensor in SENSORS else None

    return state


# ============================================================
# 화면 렌더링
# ============================================================

@app.callback(
    [
        Output(
            f"page-{i}",
            "style",
        )
        for i in range(5)
    ]
    +
    [
        Output(
            f"nav-{i}",
            "className",
        )
        for i in range(5)
    ]
    +
    [
        Output(
            f"nav-{i}",
            "disabled",
        )
        for i in range(5)
    ]
    +
    [
        Output(
            "context-1-target",
            "children",
        ),
        Output(
            "context-2-target",
            "children",
        ),
        Output(
            "context-3-target",
            "children",
        ),

        Output("period", "value"),
        Output("period-overview", "value"),
        Output("period-priority", "value"),
        Output("period-diagnosis", "value"),

        Output(
            "detail-title",
            "children",
        ),

        Output(
            "detail-priority",
            "children",
        ),

        Output(
            "detail-back",
            "children",
        ),

        Output(
            "sensor-0",
            "className",
        ),
        Output(
            "sensor-1",
            "className",
        ),
        Output(
            "sensor-2",
            "className",
        ),
        Output(
            "sensor-3",
            "className",
        ),
        Output(
            "sensor-4",
            "className",
        ),

        Output(
            "sensor-chart",
            "children",
        ),

        Output(
            "response-context",
            "children",
        ),
    ],
    Input(
        "app-state",
        "data",
    ),
    Input("prediction-results", "data"),
    Input("inspection-hours", "data"),
)
def render_state(state, results, inspection_hours):

    page = state["page"]

    page_styles = [
        {
            "display":
            "block"
            if i == page
            else "none"
        }
        for i in range(5)
    ]

    nav_classes = [
        "active"
        if i == page
        else ""
        for i in range(5)
    ]

    nav_disabled = []

    for i in range(5):

        disabled = False

        if i > 0 and not state["done"]:
            disabled = True

        if (
            state["scope"] != "all"
            and i in [1, 2]
        ):
            disabled = True

        nav_disabled.append(
            disabled
        )

    unit_name = (
        UNITS[state["unit"]]["id"]
    )

    scope_name = (
        "전체설비"
        if state["scope"] == "all"
        else unit_name
    )

    result = (results or {}).get(unit_name, {})
    timestamp = result.get("timestamp")
    context_target = html.Span(["검사 대상 ", html.B(scope_name)])
    period_control_value = (
        state["hours"]
        if inspection_hours is not None and int(inspection_hours) == int(state["hours"])
        else no_update
    )

    sensor_classes = []

    for i in range(len(SENSORS)):

        cls = "small"

        if state["sensor"] == i:
            cls += " active"

        sensor_classes.append(cls)

    sensor_name = state.get("sensor_variable") or SENSORS[state.get("sensor", 0) or 0]

    sensor_timestamp = result.get("timestamp")
    sensor_hours = int(inspection_hours if inspection_hours is not None else state.get("hours", 72))
    sensor_seconds = 10 ** 9 if sensor_hours == 999 else sensor_hours * 3600
    if sensor_hours <= 24:
        axis_divisor, axis_suffix, axis_unit = 3600, "h", "시간"
        trend_window_seconds, trend_label = 1800, "30분 추세선"
    elif sensor_hours <= 24 * 30:
        axis_divisor, axis_suffix, axis_unit = 86400, "일", "일"
        trend_window_seconds, trend_label = 10800, "3시간 추세선"
    else:
        axis_divisor, axis_suffix, axis_unit = 86400 * 30, "개월", "월"
        trend_window_seconds, trend_label = 604800, "7일 추세선"
    axis_span = sensor_seconds / axis_divisor
    try:
        points = sensor_series(unit_name, sensor_name, sensor_timestamp, seconds=sensor_seconds, limit=3000)
        if points:
            axis_span = (sensor_timestamp - points[0][0]) / axis_divisor
        fault_events = [
            row for row in filtered_faults(sensor_hours, sensor_timestamp)
            if row.get("tool_id") == unit_name
        ]
        fault_styles = {
            "pressure_low": ("#e25563", "triangle-down"),
            "pressure_high": ("#8b5cf6", "diamond"),
            "leak": ("#f08c2e", "x"),
        }
        fault_traces = []
        for fault_type in ("pressure_low", "pressure_high", "leak"):
            typed_events = [row for row in fault_events if row.get("fault_type") == fault_type]
            if not typed_events or not points:
                continue
            marker_color, marker_symbol = fault_styles[fault_type]
            nearest_values = [
                min(points, key=lambda point: abs(point[0] - row["time"]))[1]
                for row in typed_events
            ]
            fault_traces.append({
                "x": [(row["time"] - sensor_timestamp) / axis_divisor for row in typed_events],
                "y": nearest_values, "type": "scatter", "mode": "markers",
                "name": display_fault(fault_type),
                "marker": {"color": marker_color, "symbol": marker_symbol, "size": 10,
                           "line": {"color": "white", "width": 1}},
                "hovertemplate": f"{display_fault(fault_type)}<br>%{{x:.2f}}{axis_suffix}<extra></extra>",
            })
        trend_values = []
        left = 0
        rolling_sum = 0.0
        for index, (point_time, point_value) in enumerate(points):
            rolling_sum += float(point_value)
            while point_time - points[left][0] > trend_window_seconds:
                rolling_sum -= float(points[left][1])
                left += 1
            trend_values.append(rolling_sum / (index - left + 1))
        figure = {
            "data": [
                {"x": [(p[0] - sensor_timestamp) / axis_divisor for p in points], "y": [p[1] for p in points],
                 "type": "scatter", "mode": "lines", "name": "원본값",
                 "line": {"color": "rgba(148,163,184,.42)", "width": 0.8}},
                {"x": [(p[0] - sensor_timestamp) / axis_divisor for p in points], "y": trend_values,
                 "type": "scatter", "mode": "lines", "name": trend_label,
                 "line": {"color": "#075bc7", "width": 3.2}},
            ] + fault_traces,
            "layout": {"height": 285, "margin": {"l": 55, "r": 20, "t": 25, "b": 45},
                       "paper_bgcolor": "#fbfcfe", "plot_bgcolor": "#fbfcfe",
                       "xaxis": {"title": f"예측 기준시점 대비 {axis_unit}", "ticksuffix": axis_suffix},
                       "yaxis": {"title": SENSOR_LABELS[sensor_name]}, "showlegend": True,
                       "legend": {"orientation": "h", "x": 0, "y": 1.12},
                       "shapes": ([{"type": "line", "x0": 0, "x1": 0,
                                    "y0": 0, "y1": 1, "yref": "paper",
                                    "line": {"color": "#c43d4b", "dash": "dash"}}]
                                  if sensor_timestamp is not None else []),
                       "annotations": ([{"x": 0, "y": 1, "yref": "paper", "text": "예측 기준시점",
                                         "showarrow": False, "xanchor": "right", "yanchor": "bottom"}]
                                       if sensor_timestamp is not None else [])},
        }
        sensor_body = dcc.Graph(figure=figure, config={"displayModeBar": False}) if points else html.Div(
            "해당 예측시각의 실제 센서 데이터가 없습니다.", className="chart-empty")
    except Exception as exc:
        sensor_body = html.Div(f"실제 센서 데이터를 읽지 못했습니다: {type(exc).__name__}", className="chart-empty")
    sensor_chart = html.Div([
        html.Div(html.Span(SENSOR_LABELS[sensor_name], className="sensor-chart-title"), className="sensor-chart-meta"),
        sensor_body,
        html.Small(f"조회 범위 {'전체 기간' if sensor_hours == 999 else f'{sensor_hours}시간'} · {axis_unit} 단위 · 고장 유형은 기호와 색으로 구분 · 0은 예측 기준시점", className="muted"),
    ])

    detail_back_text = (
        "← 우선순위"
        if state["scope"] == "all"
        else "← 검사 설정"
    )

    priority = result.get("priority")
    priority_label = f"점검 순위 {priority}순위" if priority else "점검 순위 제외"

    database_status = (
        "운영 데이터베이스 연결됨"
        if database_is_connected()
        else "운영 데이터베이스 연결 확인 필요"
    )

    response_context = [
        html.Span(
            [
                "대상 장비 ",
                html.B(
                    unit_name,
                    className="unitname",
                ),
            ]
        ),

        html.Span(["현재 상태 ", html.B(result.get("model1", {}).get("status", "검사 전"))]),

        html.Span([
            "24시간 위험도 ",
            html.B(f'{result.get("model1", {}).get("risk_score"):.3f}'
                   if result.get("model1", {}).get("available") else "판단 보류"),
        ]),

        html.Span(["예상 유형 ", html.B(display_fault(result.get("model2", {}).get("predicted_class", "검사 전")))]),

        html.Span(
            "조치 기록은 브라우저 내 저장 · "
            + database_status
        ),
    ]

    return (
        page_styles
        + nav_classes
        + nav_disabled
        + [
            context_target,
            context_target,
            context_target,

            period_control_value,
            period_control_value,
            period_control_value,
            period_control_value,

            f"{unit_name} 상세진단",

            priority_label,

            detail_back_text,

            sensor_classes[0],
            sensor_classes[1],
            sensor_classes[2],
            sensor_classes[3],
            sensor_classes[4],

            sensor_chart,

            response_context,
        ]
    )


# ============================================================
# 보고서 저장
# ============================================================

@app.callback(
    [
        Output(
            "reports-store",
            "data",
        ),
        Output(
            "save-error",
            "children",
        ),
        Output(
            "issue",
            "value",
        ),
        Output(
            "action",
            "value",
        ),
    ],
    Input(
        "save-report",
        "n_clicks",
    ),
    [
        State(
            "reports-store",
            "data",
        ),
        State(
            "app-state",
            "data",
        ),
        State(
            "worker",
            "value",
        ),
        State(
            "report-date",
            "value",
        ),
        State(
            "report-status",
            "value",
        ),
        State(
            "issue",
            "value",
        ),
        State(
            "action",
            "value",
        ),
    ],
    prevent_initial_call=True,
)
def save_report(
    n_clicks,
    reports,
    state,
    worker,
    report_date,
    status,
    issue,
    action,
):

    if not worker or not worker.strip():
        return (
            no_update,
    ALL,
            "작업자를 입력해 주세요.",
            no_update,
    ALL,
            no_update,
    ALL,
        )

    if not issue or not issue.strip():

        return (
            no_update,
    ALL,
            "이상 상황을 입력해 주세요.",
            no_update,
    ALL,
            no_update,
    ALL,
        )

    if not action or not action.strip():

        return (
            no_update,
    ALL,
            "대응 내용을 입력해 주세요.",
            no_update,
    ALL,
            no_update,
    ALL,
        )

    reports = reports or []

    unit_name = (
        UNITS[state["unit"]]["id"]
    )

    new_report = {
        "unit": unit_name,
        "name": worker.strip(),
        "date": report_date,
        "status": status,
        "issue": issue.strip(),
        "action": action.strip(),
    }

    create_system_log(
    admin_id=current_admin_id(),
    event_type="REPORT_CREATE",
    event_level="INFO",
    tool_id=unit_name,
    message="설비 조치 보고서 작성",
    details={
        "report_date": report_date,
        "report_status": status,
    },
)


    reports = [
        new_report,
        *reports,
    ]

    return (
        reports,
        "",
        "",
        "",
    )


# ============================================================
# 보고서 표시
# ============================================================

@app.callback(
    Output(
        "report-list",
        "children",
    ),
    [
        Input(
            "reports-store",
            "data",
        ),
        Input(
            "app-state",
            "data",
        ),
    ],
)
def render_reports(
    reports,
    state,
):

    reports = reports or []

    unit_name = (
        UNITS[state["unit"]]["id"]
    )

    filtered = [
        r
        for r in reports
        if r.get("unit")
        == unit_name
    ]

    if not filtered:

        return html.Div(
            "(대응 기록 목록 공간)",
            className="empty",
        )

    cards = []

    for report in filtered:

        status_class = (
            "normal"
            if report["status"]
            == "조치 완료"
            else "unknown"
        )

        cards.append(
            html.Article(
                className="report",
                children=[

                    html.Div(
                        className="reporttop",
                        children=[
                            html.B(
                                report["name"]
                            ),

                            html.Span(
                                report[
                                    "status"
                                ],
                                className=(
                                    "badge "
                                    + status_class
                                ),
                            ),
                        ],
                    ),

                    html.Small(
                        f'{report["date"]} · '
                        f'{report["unit"]}'
                    ),

                    html.P(
                        html.B(
                            report["issue"]
                        )
                    ),

                    html.P(
                        report["action"]
                    ),
                ],
            )
        )

    return cards


# ============================================================
# Guide Modal
# ============================================================

@app.callback(
    Output(
        "guide-modal",
        "style",
    ),
    [
        Input(
            "flowBtn",
            "n_clicks",
        ),
        Input(
            "close-guide",
            "n_clicks",
        ),
    ],
    prevent_initial_call=True,
)
def toggle_guide(
    open_click,
    close_click,
):

    if ctx.triggered_id == "flowBtn":

        return {
            "display": "flex"
        }

    return {
        "display": "none"
    }


def risk_bar(model1):
    if not model1["available"]:
        return html.Span("판단 보류", className="muted")
    score = model1["risk_score"]
    tone = {"정상": "normal", "주의": "warning", "위험": "danger"}.get(model1.get("status"), "unknown")
    return html.Div([
        html.Span(f"{score:.3f}", className="risk-number"),
        html.Span(html.Span(style={"width": f"{max(0, min(score, 1)) * 100:.1f}%"}), className=f"risk-bar {tone}"),
    ], className="risk-cell")


def status_distribution(results):
    ordered = sorted(results.items(), key=lambda pair: pair[0])
    boxes = []
    for tool, result in ordered:
        model1 = result["model1"]
        status = model1["status"]
        tone = {"정상": "normal", "주의": "warning", "위험": "danger"}.get(status, "unknown")
        available = model1["available"]
        risk_score = float(model1["risk_score"]) if available else 0.0
        risk_label = f"24시간 내 위험도 {risk_score:.3f}" if available else "24시간 내 위험도 판단 보류"
        boxes.append(html.Button([
            html.Span(tool, className="unit-status-name"),
            html.Span(
                html.Span(
                    className=f"unit-status-risk-fill {tone}",
                    style={"width": f"{max(0, min(risk_score, 1)) * 100:.1f}%"},
                ),
                className="unit-status-risk-track",
                role="img",
                **{"aria-label": risk_label},
            ),
        ], id={"type": "detail-btn", "tool": tool, "view": "overview"}, className=f"unit-status-box {tone}"))
    return html.Section([
        html.Div([
            html.Div([
                html.H2("전체 설비 상태 분포"),
                html.Div([
                    html.Span("고장위험도", className="unit-status-risk-label"),
                    html.Span(
                        className="unit-status-risk-empty-bar",
                        role="img",
                        **{"aria-label": "고장위험도 바"},
                    ),
                ], className="unit-status-header-legend"),
            ], className="unit-status-title-group"),
        ], className="section-title"),
        html.P("각 설비 클릭 시 설비별 상세진단을 확인할 수 있습니다.", className="unit-click-hint"),
        html.Div(boxes, className="unit-status-grid"),
    ], className="panel status-overview")


def operation_summary(tool, timestamp):
    rows = sensor_series(tool, SENSORS[0], timestamp, seconds=86400, limit=2000)
    intervals = []
    for a, b in zip(rows, rows[1:]):
        delta = b[0] - a[0]
        if delta <= 0:
            continue
        # 관측 행이 이어지는 최대 120초는 가동, 그 이후의 공백만 비가동으로 집계
        running_part = min(delta, 120)
        if running_part:
            intervals.append((running_part, "running"))
        if delta > 120:
            intervals.append((delta - 120, "stopped"))
    def fmt_time(seconds):
        minutes = round(seconds / 60)
        return f"{minutes}m" if minutes < 60 else f"{minutes / 60:.1f}h"
    grouped = []
    for seconds, state in intervals:
        if grouped and grouped[-1][1] == state:
            grouped[-1] = (grouped[-1][0] + seconds, state)
        else:
            grouped.append((seconds, state))
    # 1분 미만 구간은 앞 구간에 흡수한 뒤 같은 상태끼리 다시 연결
    normalized = []
    for seconds, state in grouped:
        if seconds < 60 and normalized:
            previous_seconds, previous_state = normalized[-1]
            normalized[-1] = (previous_seconds + seconds, previous_state)
        elif normalized and normalized[-1][1] == state:
            normalized[-1] = (normalized[-1][0] + seconds, state)
        else:
            normalized.append((seconds, state))
    grouped = []
    for seconds, state in normalized:
        if grouped and grouped[-1][1] == state:
            grouped[-1] = (grouped[-1][0] + seconds, state)
        else:
            grouped.append((seconds, state))
    running = sum(seconds for seconds, state in grouped if state == "running")
    stopped = sum(seconds for seconds, state in grouped if state == "stopped")
    total = running + stopped
    if not total:
        return html.Div([html.H3("24시간 가동 상태"), html.P("가동 이력을 확인할 수 없습니다.", className="muted")], className="detail-facts")
    # 실제 시간순에 맞춰 가동/비가동 구간을 바에서도 각각 분리
    timeline = [
        html.Span(
            fmt_time(seconds) if seconds / total >= 0.035 else "",
            style={"width": f"{seconds / total * 100:.3f}%"},
            className=f"operation-segment {state}",
            title=f"{'가동' if state == 'running' else '비가동'} {fmt_time(seconds)}",
        )
        for seconds, state in grouped
    ]
    return html.Div([
        html.H3("24시간 가동 현황"),
        html.Div(timeline, className="operation-bar timeline-bar"),
        html.Div([
            html.Div([
                html.Span("가동", className="operation-name"),
                html.Span("–", className="operation-separator"),
                html.B(fmt_time(running)),
            ], className="operation-label running"),
            html.Div([
                html.Span("비가동", className="operation-name"),
                html.Span("–", className="operation-separator"),
                html.B(fmt_time(stopped)),
            ], className="operation-label stopped"),
        ], className="operation-legend"),
        html.P("원본 센서 관측 간격 기준 · 120초 초과 공백은 비가동으로 집계", className="muted compact"),
    ], className="detail-facts operation-summary")


def risk_meter(model1, model2=None):
    if not model1["available"]:
        return html.Div([
            html.Div("판단 보류", className="risk-score-large"),
            html.P(model1.get("reason", "센서 데이터 또는 실행 환경을 확인해 주세요."), className="muted"),
        ], className="risk-meter-card")
    score = model1["risk_score"]
    status_tone = {"정상": "normal", "주의": "warning", "위험": "danger"}.get(model1["status"], "unknown")
    fault_value = (model2 or {}).get("predicted_class")
    fault_tone = {
        "pressure_low": "pressure-low", "Low Pressure": "pressure-low",
        "pressure_high": "pressure-high", "High Pressure": "pressure-high",
        "leak": "leak", "Leak": "leak",
    }.get(fault_value, "unknown")
    fault_symbol = {"pressure-low": "↓", "pressure-high": "↑", "leak": "◇"}.get(fault_tone, "·")
    return html.Div([
        html.Div([
            html.Div([
                html.Span(className="risk-zone normal-zone"),
                html.Span(className="risk-zone warning-zone"),
                html.Span(className="risk-zone danger-zone"),
                html.I(style={"left": f"{max(1, min(score * 100, 99)):.1f}%"}),
            ], className="risk-meter"),
        ]),
        html.Div([
            html.Div([
                html.P("현재 상태", className="muted"),
                html.Strong(model1["status"], className=f"status-text {status_tone}"),
            ], className=f"detail-core-card status-state-card {status_tone}"),
            html.Div([html.P("예상 고장 유형", className="muted"), html.B([html.Span(fault_symbol, className="fault-symbol"), display_fault(fault_value)], className="detail-core-value")], className=f"detail-core-card predicted-fault-card {fault_tone}"),
        ], className="detail-core-grid"),
    ], className="risk-meter-card")


def result_table(results, view, sort_mode="risk"):
    rows = []
    fault_order = {
        "pressure_low": 0, "Low Pressure": 0,
        "pressure_high": 1, "High Pressure": 1,
        "leak": 2, "Leak": 2,
    }
    if sort_mode == "fault":
        ordered = sorted(results.items(), key=lambda pair: (
            fault_order.get(pair[1].get("model2", {}).get("predicted_class"), 9),
            pair[1].get("priority") is None,
            pair[1].get("priority") or 999,
            pair[0],
        ))
    elif sort_mode == "history":
        ordered = sorted(results.items(), key=lambda pair: (
            -(pair[1].get("fault_count") if pair[1].get("fault_count") is not None else -1),
            pair[1].get("priority") is None,
            pair[1].get("priority") or 999,
            pair[0],
        ))
    else:
        ordered = sorted(results.items(), key=lambda pair: (
            pair[1].get("priority") is None,
            pair[1].get("priority") or 999,
            pair[0],
        ))
    for tool, result in ordered:
        model1, model2 = result["model1"], result["model2"]
        score = f'{model1["risk_score"]:.3f}' if model1["available"] else "판단 보류"
        status = model1["status"]
        badge = {"정상": "normal", "주의": "warning", "위험": "danger"}.get(status, "unknown")
        fault_value = model2.get("predicted_class")
        fault_badge = {
            "pressure_low": "pressure-low", "Low Pressure": "pressure-low",
            "pressure_high": "pressure-high", "High Pressure": "pressure-high",
            "leak": "leak", "Leak": "leak",
        }.get(fault_value, "unknown")
        rows.append(html.Tr([
            html.Td(result.get("priority") or "—"), html.Td(tool),
            html.Td(html.Span(status, className=f"badge {badge}")), html.Td(risk_bar(model1)),
            html.Td(html.Span(display_fault(fault_value), className=f"fault-type-badge {fault_badge}", title=(
                "예상 고장 유형은 저장된 Model 2 검증 결과이며, "
                "24시간 내 고장 위험도와 예측 기준시각이 다를 수 있습니다."
                if result.get("data_mode") == "replay" else ""))),
            html.Td(result.get("fault_count") if result.get("fault_count") is not None else "—"),
            html.Td(html.Button("상세진단 ↗", id={"type": "detail-btn", "tool": tool, "view": view}, className="small")),
        ], className=f"priority-row status-{badge}"))
    headings = ["순위", "설비", "상태", "24시간 내 고장 위험도", "예상 고장 유형 ⓘ", "기간 내 고장", "상세"]
    return html.Table([html.Thead(html.Tr([html.Th(h) for h in headings])), html.Tbody(rows)])


def fault_visuals(hours, tool=None, cutoff=REFERENCE_TIME):
    cutoff = int(cutoff or REFERENCE_TIME)
    rows = filtered_faults(hours, cutoff, tool)
    by_type = {"pressure_low": 0, "pressure_high": 0, "leak": 0}
    for row in rows:
        by_type[row["fault_type"]] = by_type.get(row["fault_type"], 0) + 1
    selected_hours = int(hours)
    if selected_hours == 999:
        start = min((row["time"] for row in filtered_faults(999, cutoff, tool)), default=cutoff)
        bin_seconds = 30 * 86400
    else:
        start = cutoff - selected_hours * 3600
        if selected_hours <= 72:
            bin_seconds = 3600
        elif selected_hours <= 720:
            bin_seconds = 86400
        else:
            bin_seconds = 7 * 86400
    bin_count = max(1, (cutoff - start + bin_seconds - 1) // bin_seconds)
    counts = [0] * bin_count
    for row in rows:
        index = min(bin_count - 1, max(0, (row["time"] - start) // bin_seconds))
        counts[index] += 1
    relative_labels = [relative_time((bin_count - 1 - i) * bin_seconds) for i in range(bin_count)]
    axis = list(range(bin_count))
    tick_step = max(1, bin_count // 6)
    tick_values = list(range(0, bin_count, tick_step))
    if bin_count - 1 not in tick_values:
        tick_values.append(bin_count - 1)
    common = {"height": 230 if tool is None else 270, "margin": {"l": 55, "r": 20, "t": 16, "b": 48},
              "paper_bgcolor": "white", "plot_bgcolor": "white", "showlegend": False,
              "font": {"size": 14}}
    if tool is not None:
        occurred = [(i, count) for i, count in enumerate(counts) if count > 0]
        occurred_x = [item[0] for item in occurred]
        occurred_y = [item[1] for item in occurred]
        max_occurred_count = max(occurred_y, default=0)
        y_tick_target = max_occurred_count / 5
        y_tick_step = next((step for step in (1, 2, 5, 10) if y_tick_target <= step), 10)
        occurred_labels = [relative_labels[i] for i in occurred_x]
        occurred_tick_step = max(1, len(occurred_x) // 6)
        occurred_tick_positions = list(range(0, len(occurred_x), occurred_tick_step))
        if occurred_x and len(occurred_x) - 1 not in occurred_tick_positions:
            occurred_tick_positions.append(len(occurred_x) - 1)
        trend = {"data": ([{"x": occurred_x, "y": occurred_y, "customdata": occurred_labels,
                            "hovertemplate": "%{customdata}<br>%{y}건<extra></extra>",
                            "type": "scatter", "mode": "lines+markers",
                            "line": {"color": "#1f6feb", "width": 2.5},
                            "marker": {"color": "#1f6feb", "size": 9}}] if occurred else []),
                 "layout": {**common,
                            "xaxis": {"title": "분석 기준시점 대비",
                                      "tickmode": "array",
                                      "tickvals": [occurred_x[i] for i in occurred_tick_positions],
                                      "ticktext": [occurred_labels[i] for i in occurred_tick_positions],
                                      "tickangle": -30, "automargin": True},
                            "yaxis": {"title": "고장 건수", "rangemode": "tozero",
                                      "dtick": y_tick_step},
                            "annotations": ([] if occurred else [{"text": "선택 기간 내 고장 이력이 없습니다.",
                                                                    "xref": "paper", "yref": "paper",
                                                                    "x": 0.5, "y": 0.5, "showarrow": False}])}}
    else:
        trend = {"data": [{"x": axis, "y": counts, "customdata": relative_labels,
                           "hovertemplate": "%{customdata}<br>%{y}건<extra></extra>",
                           "type": "scatter", "mode": "lines+markers", "line": {"color": "#1f6feb"}}],
                 "layout": {**common, "xaxis": {"title": "분석 기준시점 대비",
                                                   "tickmode": "array", "tickvals": tick_values,
                                                   "ticktext": [relative_labels[i] for i in tick_values]},
                            "yaxis": {"title": "고장 건수", "rangemode": "tozero"}}}
    overview_compact = tool is None
    types = {"data": [{
        "labels": ["냉각 압력 저하", "냉각 압력 과다", "냉각수 누수"],
        "values": [by_type["pressure_low"], by_type["pressure_high"], by_type["leak"]],
        "type": "pie",
        "hole": 0.58,
        "textinfo": "percent",
        "textposition": "inside",
        "textfont": {"size": 16, "color": "white"},
        "hovertemplate": "%{label}<br>%{value}건 · %{percent}<extra></extra>",
        "marker": {"colors": [FAULT_TYPE_COLORS[key] for key in ("pressure_low", "pressure_high", "leak")],
                   "line": {"color": "#ffffff", "width": 4}},
        "pull": [0.015, 0.015, 0.015],
    }], "layout": {**common, "showlegend": not overview_compact,
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "legend": {"orientation": "h", "y": -0.12}}}
    return rows, trend, types


@app.callback(
    Output("prediction-results", "data"),
    Output("inspection-status", "children"),
    Output("inspection-hours", "data"),
    Input("start-btn", "n_clicks"),
    Input("period-overview", "value"),
    Input("period-priority", "value"),
    Input("period-diagnosis", "value"),
    State("scope", "value"),
    State("period", "value"),
    State("app-state", "data"),
    State("inspection-hours", "data"),
    prevent_initial_call=True,
    running=[(Output("start-btn", "disabled"), True, False),
             (Output("scope", "disabled"), True, False),
             (Output("period", "disabled"), True, False),
             (Output("period-overview", "disabled"), True, False),
             (Output("period-priority", "disabled"), True, False),
             (Output("period-diagnosis", "disabled"), True, False),
             (Output("analysis-loading", "style"), {"display": "grid"}, {"display": "none"})],
)
def run_inspection(
    clicks,
    overview_period,
    priority_period,
    diagnosis_period,
    scope,
    period,
    state,
    inspection_hours,
):

    trigger = ctx.triggered_id
    if trigger == "start-btn":
        period = int(period)
    else:
        period = int({
            "period-overview": overview_period,
            "period-priority": priority_period,
            "period-diagnosis": diagnosis_period,
        }[trigger])
        if not state.get("done") or state.get("page") == 0:
            return no_update, no_update, no_update
        if inspection_hours is not None and int(inspection_hours) == period:
            return no_update, no_update, no_update

    admin_id = current_admin_id()

    selected = (
        TOOLS
        if scope == "all"
        else [TOOLS[int(scope)]]
    )

    # ========================================================
    # 검사 시작 로그
    # ========================================================

    create_system_log(
        admin_id=admin_id,
        event_type="INSPECTION_START",
        event_level="INFO",
        message="설비 상태 조회 시작",
        details={
            "scope": scope,
            "period_hours": int(period),
            "equipment_count": len(selected),
        },
    )

    try:

        # ====================================================
        # 기존 Model 1 / Model 2 추론
        # 기존 계산 방식 그대로 유지
        # ====================================================

        results = inspect_tools(
            selected,
            int(period),
        )

        # 기존 Replay fallback 유지
        if not any(
            r["model1"]["available"]
            for r in results.values()
        ):

            results = replay_results(
                selected,
                int(period),
            )

        else:

            for result in results.values():
                result["data_mode"] = (
                    "inference"
                )

        # ====================================================
        # 기존 고장 이력 계산
        # ====================================================

        for tool, result in results.items():
            cutoff = int(result.get("timestamp") or REFERENCE_TIME)
            tool_history = filtered_faults(int(period), cutoff, tool)
            result["fault_count"] = len(tool_history)
            latest_fault = max((row["time"] for row in tool_history), default=None)
            result["seconds_since_fault"] = cutoff - latest_fault if latest_fault is not None else None

        # ====================================================
        # 주의 / 위험 로그
        # ====================================================

        for tool, result in results.items():

            model1 = result.get(
                "model1",
                {},
            )

            if not model1.get(
                "available"
            ):
                continue

            status = model1.get(
                "status"
            )

            if status not in (
                "주의",
                "위험",
            ):
                continue

            model2 = result.get(
                "model2",
                {},
            )

            create_system_log(
                admin_id=admin_id,

                event_type=(
                    "RISK_DANGER"
                    if status == "위험"
                    else "RISK_CAUTION"
                ),

                event_level=(
                    "ERROR"
                    if status == "위험"
                    else "WARNING"
                ),

                tool_id=tool,

                message=(
                    f"{status} 상태 설비 감지"
                ),

                details={
                    "risk_score": float(
                        model1[
                            "risk_score"
                        ]
                    ),

                    "status": status,

                    "predicted_fault_type":
                    (
                        model2.get(
                            "predicted_class"
                        )
                        if model2.get(
                            "available"
                        )
                        else None
                    ),

                    "data_mode":
                    result.get(
                        "data_mode"
                    ),
                },
            )

        # ====================================================
        # 완료 정보
        # ====================================================

        available = sum(
            r["model1"]["available"]
            for r in results.values()
        )

        caution_count = sum(
            r["model1"].get(
                "status"
            )
            == "주의"
            for r in results.values()
        )

        danger_count = sum(
            r["model1"].get(
                "status"
            )
            == "위험"
            for r in results.values()
        )

        replay_used = any(
            r.get("data_mode")
            == "replay"
            for r in results.values()
        )

        create_system_log(
            admin_id=admin_id,
            event_type=(
                "INSPECTION_COMPLETE"
            ),
            event_level="INFO",
            message="설비 상태 조회 완료",
            details={
                "equipment_count":
                    len(results),

                "available_count":
                    available,

                "deferred_count":
                    len(results)
                    - available,

                "caution_count":
                    caution_count,

                "danger_count":
                    danger_count,

                "data_mode":
                    (
                        "replay"
                        if replay_used
                        else "inference"
                    ),
            },
        )

    except Exception as exc:

        # ====================================================
        # 검사 오류 로그
        # ====================================================

        create_system_log(
            admin_id=admin_id,
            event_type=(
                "INSPECTION_ERROR"
            ),
            event_level="ERROR",
            message=(
                "설비 상태 조회 중 오류 발생"
            ),
            details={
                "error_type":
                    type(exc).__name__,
            },
        )

        raise

    mode = (
        " · 저장된 실제 검증 결과 재생"
        if replay_used
        else ""
    )

    return (
        results,

        (
            f"검사 완료: "
            f"{len(results)}대 · "
            f"위험도 판정 "
            f"{available}대 · "
            f"판단 보류 "
            f"{len(results) - available}대"
            f"{mode}"
        ),
        period,
    )


@app.callback(
    Output("overview-kpis", "children"), Output("overview-results", "children"),
    Output("priority-results", "children"), Output("detail-status", "children"),
    Output("detail-history", "children"), Output("detail-errors", "children"),
    Output("recommended-actions", "children"),
    Output("forecast-results", "children"), Output("overview-fault-trend", "figure"),
    Output("overview-fault-types", "figure"), Output("detail-fault-trend", "figure"),
    Output("detail-fault-types", "figure"), Output("detail-fault-list", "children"),
    Input("prediction-results", "data"), Input("app-state", "data"),
    Input("inspection-hours", "data"),
    Input("priority-sort", "value"),
)
def render_predictions(results, state, inspection_hours, priority_sort):
    results = results or {}
    active_hours = int(inspection_hours if inspection_hours is not None else state.get("hours", 72))
    counts = {status: sum(r["model1"]["status"] == status for r in results.values()) for status in ("정상", "주의", "위험", "판단 보류")}
    kpis = [
        kpi("검사 설비", value=f"{len(results)}대"),
        kpi("정상", cls="green", value=f'{counts["정상"]}대'),
        kpi("주의", cls="amber", value=f'{counts["주의"]}대'),
        kpi("위험", cls="red", value=f'{counts["위험"]}대'),
    ]
    overview = status_distribution(results)
    tool = TOOLS[state["unit"]]
    result = results.get(tool)
    fault_rows_all, overview_trend, overview_types = fault_visuals(active_hours)
    detail_cutoff = int((result or {}).get("timestamp") or REFERENCE_TIME)
    detail_faults, detail_trend, detail_types = fault_visuals(active_hours, tool, detail_cutoff)
    fault_list = html.Div("선택 기간 내 고장 이력이 없습니다.", className="empty") if not detail_faults else html.Table([
        html.Thead(html.Tr([html.Th("발생 시점"), html.Th("고장 유형")])),
        html.Tbody([html.Tr([html.Td(relative_time(detail_cutoff - row["time"])),
                            html.Td(display_fault(row["fault_type"]))])
                    for row in sorted(detail_faults, key=lambda x: x["time"], reverse=True)])
    ])
    if not result:
        return (kpis, overview, result_table(results, "priority", priority_sort), "검사 전", [], "", build_recommended_actions(None), "검사 전",
                overview_trend, overview_types, detail_trend, detail_types, fault_list)
    m1, m2 = result["model1"], result["model2"]
    timestamp = result.get("timestamp")
    detail = [risk_meter(m1, m2), operation_summary(tool, timestamp)]
    age = result.get("seconds_since_fault")
    history = [kpi("조회기간 내 고장 횟수", value=result.get("fault_count") if result.get("fault_count") is not None else "조회 불가"),
               kpi("마지막 고장 이후 경과", value=f"{age / 3600:.1f}시간" if age is not None else ("조회 불가" if result.get("fault_count") is None else "이력 없음"))]
    errors = [html.P(f'{name}: {model["error"]}') for name, model in (("Model 1", m1), ("Model 2", m2)) if not model["available"]]
    recommended_actions = build_recommended_actions(m2)
    if result.get("data_mode") == "replay":
        errors.append(html.P("저장된 실제 검증 결과 재생 · 설비별 관측 시점이 다릅니다. 위험도는 실제 발생확률이 아닙니다. 예상 고장 유형은 별도 저장 검증 사례입니다."))
    type_counts = {"pressure_low": 0, "pressure_high": 0, "leak": 0}
    for row in fault_rows_all:
        type_counts[row["fault_type"]] += 1
    total_faults = len(fault_rows_all)
    fault_legend = [
        ("pressure_low", "냉각 압력 저하"),
        ("pressure_high", "냉각 압력 과다"),
        ("leak", "냉각수 누수"),
    ]
    forecast = html.Div([
        html.Div([html.Strong(f"총 {total_faults}건"), html.Span("선택 기간 실제 발생 기록")], className="fault-total"),
        html.Div([
            html.Div([
                html.Span(className="fault-dot", style={"backgroundColor": FAULT_TYPE_COLORS[key]}),
                html.Span(label, className="fault-legend-label"),
                html.Strong(f"{type_counts[key]}건"),
                html.Span(f"{type_counts[key] / total_faults * 100:.0f}%" if total_faults else "0%", className="fault-percent"),
            ], className="fault-legend-row")
            for key, label in fault_legend
        ], className="fault-legend"),
    ])
    return (kpis, overview, result_table(results, "priority", priority_sort), detail, history, errors, recommended_actions, forecast,
            overview_trend, overview_types, detail_trend, detail_types, fault_list)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8063,
        debug=False
    )
