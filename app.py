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
)


# ============================================================
# APP
# ============================================================

app = Dash(__name__)
server = app.server


# ============================================================
# 기본 데이터
# 실제 모델 연결 전 placeholder
# ============================================================

UNITS = [
    {
        "id": "(장비명)",
        "ttf": None,
        "type": "",
    }
]

SENSORS = [
    "냉각 압력",
    "냉각 유량",
    "공정 전류",
]

STEP_NAMES = [
    "검사",
    "전체 결과",
    "우선순위",
    "상세진단",
    "대응·피드백",
]


# ============================================================
# Helper Components
# ============================================================

def kpi(label, note="", cls=""):
    return html.Div(
        className=f"kpi {cls}",
        children=[
            html.Div(label, className="label"),
            html.Div("(수치 영역)", className="metric-placeholder"),
            html.Div(note, className="note"),
        ],
    )


def placeholder(text, extra_class=""):
    return html.Div(
        text,
        className=f"placeholder {extra_class}",
    )


def page_head(number, title, subtitle, action=None):
    children = [
        html.Div(
            [
                html.Span(number, className="eyebrow"),
                html.H1(title),
                html.P(subtitle, className="muted"),
            ]
        )
    ]

    if action is not None:
        children.append(action)

    return html.Div(
        children,
        className="pagehead",
    )


def context_bar(context_id):
    return html.Div(
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
                html.Span("F", className="logo"),
                html.Span(
                    [
                        "FlowCool ",
                        html.Small(
                            "TEAM 06 / 냉각설비 예지보전"
                        ),
                    ]
                ),
            ],
        ),

        html.Div(
            className="header-right",
            children=[
                html.Span(
                    "DASH APP",
                    className="demo",
                ),

                html.Button(
                    "사용 가이드 ↗",
                    id="flowBtn",
                    className="quiet",
                ),
            ],
        ),
    ]
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
            "01",
            "냉각설비 검사",
            "검사할 설비와 조회 기간을 선택하세요.",
        ),

        html.Div(
            className="setup",
            children=[

                html.Section(
                    className="panel",
                    children=[

                        html.H2("검사 조건"),

                        html.Div(
                            className="field",
                            children=[
                                html.Label(
                                    "검사 대상 장비"
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
                                ),
                            ],
                        ),

                        html.Div(
                            className="field",
                            children=[
                                html.Label("검사 기간"),

                                dcc.Dropdown(
                                    id="period",
                                    options=[
                                        {
                                            "label": "최근 12시간",
                                            "value": 12,
                                        },
                                        {
                                            "label": "최근 24시간",
                                            "value": 24,
                                        },
                                        {
                                            "label": "최근 3일",
                                            "value": 72,
                                        },
                                        {
                                            "label": "전체 기간",
                                            "value": 999,
                                        },
                                    ],
                                    value=72,
                                    clearable=False,
                                ),

                                html.Span(
                                    "조회할 기간을 선택하세요.",
                                    className="muted",
                                ),
                            ],
                        ),

                        html.Button(
                            [
                                "검사 시작 ",
                                html.Span("→"),
                            ],
                            id="start-btn",
                            className="primary",
                        ),

                        html.P(
                            "센서 데이터와 고장 이력을 "
                            "확인한 후 예측 결과를 제공합니다.",
                            className="muted",
                        ),
                    ],
                ),

                html.Aside(
                    className="blueprint",
                    children=[

                        html.Span(
                            "INSPECTION PIPELINE",
                            className="eyebrow",
                        ),

                        html.H2(
                            "검사부터 대응까지",
                            style={
                                "marginTop": "16px"
                            },
                        ),

                        html.Div(
                            [
                                html.B("01"),
                                html.Span(
                                    "대상·기간 설정"
                                ),
                            ],
                            className="path",
                        ),

                        html.Div(
                            [
                                html.B("02"),
                                html.Span(
                                    "고장 이력 & 예측 결과 확인"
                                ),
                            ],
                            className="path",
                        ),

                        html.Div(
                            [
                                html.B("03"),
                                html.Span(
                                    "우선 관리 대상 선정"
                                ),
                            ],
                            className="path",
                        ),

                        html.Div(
                            [
                                html.B("04"),
                                html.Span(
                                    "설비별 상세진단"
                                ),
                            ],
                            className="path",
                        ),

                        html.Div(
                            [
                                html.B("05"),
                                html.Span(
                                    "대응 기록 & 근무자 인계"
                                ),
                            ],
                            className="path",
                        ),

                        html.Div(
                            className="setupnote",
                            children=[
                                html.P(
                                    "단일설비는 상세진단으로 "
                                    "바로 이동합니다."
                                ),

                                html.P(
                                    [
                                        "현재 차트와 결과는 "
                                        "자리표시자입니다.",
                                        html.Br(),
                                        "추후 실제 예측 모델과 연결합니다.",
                                    ],
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
            "02",
            "전체 검사 결과",
            "과거 고장 이력과 앞으로의 "
            "점검 대상을 함께 확인합니다.",
            html.Button(
                "우선순위 보기 →",
                id="to-priority",
                className="primary",
            ),
        ),

        context_bar("context-1"),

        html.Div(
            className="section-title",
            children=[
                html.H2("조회기간 내 결과")
            ],
        ),

        html.Div(
            className="kpis",
            children=[
                kpi("검사 대상"),
                kpi(
                    "이상 장비 수",
                    cls="red",
                ),
                kpi("총 고장 발생 건수"),
                kpi(
                    "예측 확인 필요",
                    cls="amber",
                ),
            ],
        ),

        html.Div(
            className="section-title",
            children=[
                html.H2("예측 결과")
            ],
        ),

        html.Div(
            className="kpis",
            children=[
                kpi("12시간 내 고장 예측"),
                kpi("24시간 내 고장 예측"),
                kpi("주의"),
                kpi("정상"),
            ],
        ),

        html.Div(
            className="section-title",
            children=[
                html.H2("기간조회 결과 리포트")
            ],
        ),

        html.Div(
            className="two",
            children=[

                html.Section(
                    className="panel",
                    children=[
                        html.H3(
                            "시간대별 고장 발생 추이"
                        ),
                        placeholder(
                            "(선 그래프 공간)"
                        ),
                    ],
                ),

                html.Section(
                    className="panel",
                    children=[
                        html.H3("고장 유형 구성"),
                        placeholder(
                            "(도넛 그래프 공간)"
                        ),
                    ],
                ),
            ],
        ),

        html.Section(
            className="panel forecast",
            children=[
                html.H2(
                    "24시간 내 점검 대상"
                ),
                placeholder(
                    "(막대 그래프 공간)"
                ),
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
            "03",
            "우선 관리 대상",
            "설비별 점검 순서와 "
            "상세진단을 확인합니다.",
            html.Button(
                "← 전체 결과",
                id="priority-back",
            ),
        ),

        context_bar("context-2"),

        html.Div(
            className="section-title",
            children=[
                html.H2(
                    "설비별 점검 순서"
                ),

                html.Span(
                    "정렬: 예상 TTF 오름차순",
                    className="muted",
                ),
            ],
        ),

        html.Section(
            className="panel",
            style={"padding": 0},
            children=[

                html.Div(
                    className="tablewrap",
                    children=[

                        html.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("순위"),
                                            html.Th("장비"),
                                            html.Th(
                                                "예측 상태"
                                            ),
                                            html.Th(
                                                "예상 TTF"
                                            ),
                                            html.Th(
                                                "예상 고장 유형"
                                            ),
                                            html.Th(
                                                "기간 내 고장"
                                            ),
                                            html.Th("진단"),
                                        ]
                                    )
                                ),

                                html.Tbody(
                                    [
                                        html.Tr(
                                            [
                                                html.Td(
                                                    "(순위)"
                                                ),
                                                html.Td(
                                                    "(장비명)"
                                                ),
                                                html.Td(
                                                    "(상태)"
                                                ),
                                                html.Td(
                                                    "(잔여시간)"
                                                ),
                                                html.Td(
                                                    "(고장 유형)"
                                                ),
                                                html.Td(
                                                    "(건수)"
                                                ),

                                                html.Td(
                                                    html.Button(
                                                        "상세진단 ↗",
                                                        id="detail-btn",
                                                        className="small",
                                                    )
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                ),

                placeholder(
                    "(우선순위 목록 공간)",
                    "table-placeholder",
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
            "04",
            html.Span(
                "(장비명) 상세진단",
                id="detail-title",
            ),
            "고장 이력과 센서 흐름을 바탕으로 "
            "예측 결과를 검토합니다.",
            html.Button(
                "← 우선순위",
                id="detail-back",
            ),
        ),

        context_bar("context-3"),

        html.Div(
            className="two equal",
            style={
                "marginBottom": "24px"
            },
            children=[
                kpi(
                    "조회기간 내 고장 횟수"
                ),
                kpi(
                    "마지막 고장 이후 경과"
                ),
            ],
        ),

        html.Section(
            className="detailstatus",
            children=[

                html.Div(
                    [
                        html.P(
                            "예측 상태",
                            className="muted",
                        ),
                        html.Div(
                            "(예측 상태)"
                        ),
                    ]
                ),

                html.Div(
                    [
                        html.P(
                            "예상 잔여시간 · TTF",
                            className="muted",
                        ),
                        html.Div(
                            "(잔여시간)"
                        ),
                    ]
                ),

                html.Div(
                    [
                        html.P(
                            "예상 고장 유형",
                            className="muted",
                        ),
                        html.Div(
                            "(고장 유형)"
                        ),
                    ]
                ),

                html.Div(
                    [
                        html.P(
                            "예측 기준시각",
                            className="muted",
                        ),
                        html.Div(
                            "(기준시각)"
                        ),
                    ]
                ),
            ],
        ),

        html.Div(
            className="section-title",
            children=[
                html.H2(
                    "기간조회 결과 리포트"
                )
            ],
        ),

        html.Div(
            className="two",
            children=[

                html.Section(
                    className="panel",
                    children=[
                        html.H3(
                            "고장 발생 시점"
                        ),

                        placeholder(
                            "(선 그래프 공간)"
                        ),

                        placeholder(
                            "(고장 이력 목록 공간)",
                            "table-placeholder",
                        ),
                    ],
                ),

                html.Section(
                    className="panel",
                    children=[
                        html.H3(
                            "고장 유형별 발생"
                        ),

                        placeholder(
                            "(도넛 그래프 공간)"
                        ),
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

                html.H2(
                    "센서 데이터 추이"
                ),

                html.Div(
                    className="sensorbuttons",
                    children=[
                        html.Button(
                            sensor,
                            id=f"sensor-{i}",
                            className="small",
                        )
                        for i, sensor
                        in enumerate(SENSORS)
                    ],
                ),

                html.Div(
                    id="sensor-chart",
                    className="placeholder",
                ),
            ],
        ),

        html.Div(
            className="actions",
            children=[
                html.Button(
                    "대응 및 피드백 →",
                    id="to-response",
                    className="primary",
                )
            ],
        ),
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
            "05",
            "대응 및 피드백",
            "점검 내용과 조치 결과를 기록하고 "
            "다음 근무자에게 전달합니다.",
            html.Button(
                "← 상세진단",
                id="response-back",
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
                                    "TTF 기반 우선순위"
                                ),
                                html.P("↓"),
                                html.P(
                                    "장비 상세진단 "
                                    "→ 대응·피드백"
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
                                    "대응·피드백"
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
            },
        ),

        dcc.Store(
            id="reports-store",
            storage_type="local",
            data=[],
        ),

        header,
        nav,

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
                    "FLOWCOOL SYSTEM / TEAM 06"
                ),
                html.Span(
                    "DASH APP / MODEL READY"
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
            "start-btn",
            "n_clicks",
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
            "detail-btn",
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
        State(
            "period",
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
    response_back,
    sensor0,
    sensor1,
    sensor2,
    state,
    scope,
    period,
):

    trigger = ctx.triggered_id

    state = dict(state)

    if trigger == "start-btn":

        state["scope"] = scope
        state["hours"] = int(period)

        state["unit"] = (
            0
            if scope == "all"
            else int(scope)
        )

        state["done"] = True

        state["page"] = (
            1
            if scope == "all"
            else 3
        )

    elif trigger and trigger.startswith(
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

    elif trigger == "detail-btn":
        state["unit"] = 0
        state["page"] = 3

    elif trigger == "detail-back":

        state["page"] = (
            2
            if state["scope"] == "all"
            else 0
        )

    elif trigger == "to-response":
        state["page"] = 4

    elif trigger == "response-back":
        state["page"] = 3

    elif trigger in [
        "sensor-0",
        "sensor-1",
        "sensor-2",
    ]:

        state["sensor"] = int(
            trigger.split("-")[1]
        )

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
            "context-1",
            "children",
        ),
        Output(
            "context-2",
            "children",
        ),
        Output(
            "context-3",
            "children",
        ),

        Output(
            "detail-title",
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
)
def render_state(state):

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

    period_name = (
        "전체 기간"
        if state["hours"] == 999
        else f'{state["hours"]}시간'
    )

    context_children = [
        html.Span(
            [
                "검사 대상 ",
                html.B(scope_name),
            ]
        ),

        html.Span(
            [
                "조회 기간 ",
                html.B(period_name),
            ]
        ),

        html.Span(
            [
                "예측 기준 ",
                html.B("(기준시각)"),
            ]
        ),
    ]

    sensor_classes = []

    for i in range(3):

        cls = "small"

        if state["sensor"] == i:
            cls += " active"

        sensor_classes.append(cls)

    sensor_name = SENSORS[
        state["sensor"]
    ]

    sensor_chart = (
        f"({sensor_name} · 선 그래프 공간)"
    )

    detail_back_text = (
        "← 우선순위"
        if state["scope"] == "all"
        else "← 검사 설정"
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

        html.Span(
            "현재 예측 (예측 상태)"
        ),

        html.Span(
            "브라우저 내 저장 · "
            "공용 데이터베이스 미연결"
        ),
    ]

    return (
        page_styles
        + nav_classes
        + nav_disabled
        + [
            context_children,
            context_children,
            context_children,

            f"{unit_name} 상세진단",

            detail_back_text,

            sensor_classes[0],
            sensor_classes[1],
            sensor_classes[2],

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
            "작업자를 입력해 주세요.",
            no_update,
            no_update,
        )

    if not issue or not issue.strip():

        return (
            no_update,
            "이상 상황을 입력해 주세요.",
            no_update,
            no_update,
        )

    if not action or not action.strip():

        return (
            no_update,
            "대응 내용을 입력해 주세요.",
            no_update,
            no_update,
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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True
    )