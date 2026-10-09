from pathlib import Path
p=Path('app.py'); s=p.read_text(encoding='utf-8')
s=s.replace('    no_update,','    no_update,\n    ALL,')
s=s.replace('\n\n# ============================================================\n# APP', '\n\nfrom src.dashboard_inference import TOOLS, inspect_tools\n\n# ============================================================\n# APP',1)
a=s.index('UNITS = ['); b=s.index('\nSENSORS =',a)
s=s[:a]+'UNITS = [{"id": tool} for tool in TOOLS]\n'+s[b:]
s=s.replace('# 실제 모델 연결 전 placeholder','# 저장 모델과 실제 검증 데이터 재생')
s=s.replace('def kpi(label, note="", cls=""):', 'def kpi(label, note="", cls="", value="—"):')
s=s.replace('html.Div("(수치 영역)", className="metric-placeholder")','html.Div(value, className="metric-placeholder")')
s=s.replace('"현재 차트와 결과는 "\n                                        "자리표시자입니다."','"저장된 실제 검증 데이터로 예측합니다."')
s=s.replace('"추후 실제 예측 모델과 연결합니다."','"설비별 마지막 관측 시점이 다릅니다. 실시간 데이터는 연결되지 않았습니다."')
s=s.replace('"센서 데이터와 고장 이력을 "\n                            "확인한 후 예측 결과를 제공합니다."','"선택 기간은 고장 이력 조회 범위입니다. 예측은 검증 구간의 마지막 관측 시점을 기준으로 합니다."')
s=s.replace('id="start-btn",','id="start-btn",',1)
# Status is separate from the button so Dash can show pending inference.
pos=s.index('                        html.P(\n                            "선택 기간')
s=s[:pos]+'                        dcc.Loading(html.Div(id="inspection-status", role="status")),\n\n'+s[pos:]
a=s.index('        html.Div(\n            className="kpis",',s.index('page_overview ='))
b=s.index('        html.Div(\n            className="section-title",\n            children=[\n                html.H2("기간조회 결과 리포트")',a)
s=s[:a]+'''        html.Div(id="overview-kpis", className="kpis"),
        html.P("위험도는 Model 1 score이며 실제 고장 발생확률이 아닙니다. 정상·주의·위험은 score에 따른 운영 상태입니다.", className="muted"),
        html.Div(id="overview-results", className="tablewrap"),

'''+s[b:]
s=s.replace('placeholder(\n                    "(막대 그래프 공간)"\n                )','html.Div(id="forecast-results")')
s=s.replace('"정렬: 예상 TTF 오름차순"','"정렬: Model 1 위험도 내림차순 · 판단 보류는 순위 제외"')
a=s.index('                        html.Table(',s.index('page_priority ='))
b=s.index('\n                    ],\n                ),',a)
s=s[:a]+'                        html.Div(id="priority-results"),'+s[b:]
s=s.replace('                placeholder(\n                    "(우선순위 목록 공간)",\n                    "table-placeholder",\n                ),','')
a=s.index('        html.Div(\n            className="two equal",',s.index('page_diagnosis ='))
b=s.index('        html.Div(\n            className="section-title",',a)
s=s[:a]+'''        html.Div(id="detail-history", className="two equal", style={"marginBottom": "24px"}),
        html.Section(id="detail-status", className="detailstatus"),
        html.Div(id="detail-errors", role="status", className="muted"),

'''+s[b:]
s=s.replace('"TTF 기반 우선순위"','"24시간 고장 위험도 기반 우선순위"')
s=s.replace('"(장비명) 상세진단"','"설비 상세진단"')
s=s.replace('"DASH APP / MODEL READY"','"DASH APP / 검증 데이터 재생"')
s=s.replace('        dcc.Store(\n            id="reports-store",','        dcc.Store(id="prediction-results", data={}),\n\n        dcc.Store(\n            id="reports-store",')
# Navigate once the inference result has actually arrived.
s=s.replace('Input(\n            "start-btn",\n            "n_clicks",\n        )','Input(\n            "prediction-results",\n            "data",\n        )',1)
s=s.replace('"detail-btn",\n            "n_clicks",','{"type": "detail-btn", "tool": ALL, "view": ALL},\n            "n_clicks",',1)
s=s.replace('if trigger == "start-btn":','if trigger == "prediction-results":\n\n        if not start:\n            return state',1)
s=s.replace('elif trigger and trigger.startswith(','elif isinstance(trigger, str) and trigger.startswith(',1)
s=s.replace('elif trigger == "detail-btn":\n        state["unit"] = 0', 'elif isinstance(trigger, dict) and trigger.get("type") == "detail-btn":\n        if not any(detail_btn or []):\n            return state\n        state["unit"] = TOOLS.index(trigger["tool"])',1)
s=s.replace('def render_state(state):','def render_state(state, results):',1)
needle='    Input(\n        "app-state",\n        "data",\n    ),\n)\ndef render_state'
s=s.replace(needle,'    Input(\n        "app-state",\n        "data",\n    ),\n    Input("prediction-results", "data"),\n)\ndef render_state',1)
s=s.replace('    context_children = [','    result = (results or {}).get(unit_name, {})\n    timestamp = result.get("timestamp")\n    timestamp_label = f"원본 경과 {timestamp:,}초" if timestamp is not None else "판단 보류"\n    context_children = [',1)
s=s.replace('html.B("(기준시각)")','html.B("설비별 기준시각 · 상세진단 확인" if state["scope"] == "all" and page != 3 else timestamp_label)',1)
s=s.replace('"현재 예측 (예측 상태)"','"현재 예측 " + result.get("model1", {}).get("status", "검사 전")',1)
# Name the remaining unconnected chart areas honestly.
s=s.replace('"(선 그래프 공간)"','"고장 추이 차트 미연결"').replace('"(도넛 그래프 공간)"','"고장 유형 차트 미연결"')
s=s.replace('"(고장 이력 목록 공간)"','"고장 이력 상세 목록 미연결"')
s=s.replace('f"({sensor_name} · 선 그래프 공간)"','f"{sensor_name} · 센서 추이 차트 미연결"')
s=s.replace('debug=True','debug=False')
# Add callbacks before RUN (all existing report and navigation behavior retained).
where=s.index('# ============================================================\n# RUN')
addition='''def result_table(results, view):
    rows = []
    ordered = sorted(results.items(), key=lambda pair: (pair[1].get("priority") is None, pair[1].get("priority") or 0, pair[0]))
    for tool, result in ordered:
        model1, model2 = result["model1"], result["model2"]
        score = f'{model1["risk_score"]:.3f}' if model1["available"] else "판단 보류"
        status = model1["status"]
        badge = {"정상": "normal", "주의": "warning", "위험": "danger"}.get(status, "unknown")
        rows.append(html.Tr([
            html.Td(result.get("priority") or "—"), html.Td(tool),
            html.Td(html.Span(status, className=f"badge {badge}")), html.Td(score),
            html.Td(model2["predicted_class"]),
            html.Td(result.get("fault_count") if result.get("fault_count") is not None else "—"),
            html.Td(html.Button("상세진단 ↗", id={"type": "detail-btn", "tool": tool, "view": view}, className="small")),
        ]))
    headings = ["순위", "장비", "예측 상태", "24시간 고장 위험도", "예상 고장 유형", "기간 내 고장", "진단"]
    return html.Table([html.Thead(html.Tr([html.Th(h) for h in headings])), html.Tbody(rows)])


@app.callback(
    Output("prediction-results", "data"), Output("inspection-status", "children"),
    Input("start-btn", "n_clicks"), State("scope", "value"), State("period", "value"),
    prevent_initial_call=True,
    running=[(Output("start-btn", "disabled"), True, False),
             (Output("scope", "disabled"), True, False), (Output("period", "disabled"), True, False)],
)
def run_inspection(clicks, scope, period):
    selected = TOOLS if scope == "all" else [TOOLS[int(scope)]]
    results = inspect_tools(selected, int(period))
    available = sum(r["model1"]["available"] for r in results.values())
    return results, f"검사 완료: {len(results)}대 · 위험도 판정 {available}대 · 판단 보류 {len(results) - available}대"


@app.callback(
    Output("overview-kpis", "children"), Output("overview-results", "children"),
    Output("priority-results", "children"), Output("detail-status", "children"),
    Output("detail-history", "children"), Output("detail-errors", "children"),
    Output("forecast-results", "children"),
    Input("prediction-results", "data"), Input("app-state", "data"),
)
def render_predictions(results, state):
    results = results or {}
    counts = {status: sum(r["model1"]["status"] == status for r in results.values()) for status in ("정상", "주의", "위험", "판단 보류")}
    kpis = [kpi("검사 대상 설비 수", value=len(results))] + [kpi(f"{status} 설비 수", value=count) for status, count in counts.items()]
    tool = TOOLS[state["unit"]]
    result = results.get(tool)
    if not result:
        return kpis, result_table(results, "overview"), result_table(results, "priority"), "검사 전", [], "", "검사 전"
    m1, m2 = result["model1"], result["model2"]
    timestamp = result.get("timestamp")
    fields = [("장비명", tool), ("24시간 내 고장 위험도", f'{m1["risk_score"]:.3f}' if m1["available"] else "판단 보류"),
              ("예측 상태", m1["status"]), ("점검 우선순위", f'{result["priority"]}순위 (선택 설비 기준)' if result.get("priority") else "순위 제외"),
              ("예상 고장 유형", m2["predicted_class"]),
              ("예측 기준시각", f"원본 데이터 경과 {timestamp:,}초" if timestamp is not None else "판단 보류")]
    detail = [html.Div([html.P(label, className="muted"), html.Div(value)]) for label, value in fields]
    if m2["available"]:
        detail.append(html.Small(f'모델 판단 Score: {m2["max_score"]:.2f}'))
    age = result.get("seconds_since_fault")
    history = [kpi("조회기간 내 고장 횟수", value=result.get("fault_count", "—")),
               kpi("마지막 고장 이후 경과", value=f"{age / 3600:.1f}시간" if age is not None else "이력 없음")]
    errors = [html.P(f'{name}: {model["error"]}') for name, model in (("Model 1", m1), ("Model 2", m2)) if not model["available"]]
    errors.append(html.P("검증 데이터 재생 · 설비별 관측 시점이 다릅니다. 위험도와 모델 판단 Score는 실제 발생확률이 아닙니다. 예상 고장 유형은 고장 전 24시간 내 패턴을 학습한 모델의 판단입니다."))
    forecast = html.P(f'점검 필요 설비 {counts["주의"] + counts["위험"]}대 · 주의 {counts["주의"]}대 / 위험 {counts["위험"]}대 · 판단 보류 {counts["판단 보류"]}대')
    return kpis, result_table(results, "overview"), result_table(results, "priority"), detail, history, errors, forecast


'''
s=s[:where]+addition+s[where:]
p.write_text(s,encoding='utf-8')
