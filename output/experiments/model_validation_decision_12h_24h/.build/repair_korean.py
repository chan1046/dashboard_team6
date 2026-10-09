"""Repair Korean text after artifact-tool's XLSX Unicode export corrupted it.

The numeric workbook, layout, formats, and values were authored by artifact-tool.
This repairs only damaged Unicode strings using their original intended labels.
"""

from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment


file = Path("output/experiments/model_validation_decision_12h_24h/모델검증_12h_24h_HGB_XGBoost_TTF_경보비교.xlsx")
wb = load_workbook(file)
names = ["결론", "분류 비교", "경보 조건", "설비별 오경보", "TTF 회귀"]
for sheet, name in zip(wb.worksheets, names):
    sheet.title = name

summary = wb[names[0]]
for cell, value in {
    "A2": "12h·24h 모델 검증 결과",
    "A5": "선정한 분류 모델", "B5": "XGBoost 12h / 24h",
    "A6": "TTF 회귀 모델", "B6": "선정 보류 — HGB-log·LSTM 모두 R² 음수",
    "A7": "24h 경보 후보", "B7": "점수 ≥ 0.36, 최근 5분 중 2회",
    "A8": "대시보드 연결", "B8": "아직 진행하지 않음",
    "A11": "24h 후보 지표", "B11": "값", "C11": "판단 기준", "D11": "해석",
    "A12": "고장 사건 탐지", "C12": "전체 181건", "D12": "기존 0.5 단독 112건",
    "A13": "10분 묶음 오경보", "C13": "기존 3,414건", "D13": "2,652건 증가",
    "A14": "설비당 12h 평균 오경보", "C14": "허용 약 4건", "D14": "관측분 환산 평균",
    "A15": "허용치 초과 설비", "C15": "20대 중", "D15": "운영 조건 미충족",
    "A16": "최다 설비 12h 오경보", "C16": "허용 약 4건", "D16": "운영 조건 미충족",
    "A19": "검증 한계",
    "A20": "임계값과 2/5회 조건을 선택한 검증 구간을 재사용했습니다. 이후 시간 구간의 독립 검증이 필요합니다.",
    "A21": "12h 오경보율은 관측된 1분 행을 720분으로 나눈 환산치이며 실제 근무조별 횟수와 다릅니다.",
    "A22": "고장 유형·TTF는 현재 모델 성능으로 표시하거나 운영 판단에 쓰지 않습니다.",
}.items():
    summary[cell] = value
summary.column_dimensions["A"].width = 34
summary.column_dimensions["B"].width = 46
summary.column_dimensions["C"].width = 22
summary.column_dimensions["D"].width = 23
for row in (20, 21, 22):
    if f"A{row}:D{row}" not in {str(item) for item in summary.merged_cells.ranges}:
        summary.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
    summary[f"A{row}"].alignment = Alignment(wrap_text=True, vertical="center")
    summary.row_dimensions[row].height = 33

classification = wb[names[1]]
classification["A2"] = "HGB와 XGBoost · 동일 검증 구간"
for i, value in enumerate(["모델", "예측시간(h)", "AP", "Precision", "Recall", "F1", "Balanced Acc.",
                           "탐지/전체", "평가가능 사건", "오경보(10분)", "TN", "FP", "FN", "TP"], 1):
    classification.cell(5, i).value = value
classification["A12"] = "비교 조건"
classification["A13"] = "20대 설비, 85개 입력변수, 시간순 80/20 분할, 24h purge, 동일 1,142,112개 검증 행."
classification["A14"] = "양성 비율이 달라 12h와 24h의 AP 수치 자체를 직접 우열 비교하지 않습니다."
classification.column_dimensions["A"].width = 20
for row in (13, 14):
    if f"A{row}:N{row}" not in {str(item) for item in classification.merged_cells.ranges}:
        classification.merge_cells(start_row=row, start_column=1, end_row=row, end_column=14)
    classification[f"A{row}"].alignment = Alignment(wrap_text=True, vertical="center")
    classification.row_dimensions[row].height = 31

alarm = wb[names[2]]
alarm["A2"] = "최근 5분 중 2회 경보 비교"
for i, value in enumerate(["모델", "예측시간(h)", "Threshold", "연속 조건", "탐지 건", "전체 181건 Recall",
                           "평가가능 Recall", "오경보(10분)", "설비당 12h 평균", "4건 초과 설비", "설비당 최고"], 1):
    alarm.cell(5, i).value = value
for row in range(6, 10):
    alarm.cell(row, 4).value = "단독" if row == 6 else "2/5회"
alarm["A12"] = "2/5회: 현재 포함 최근 5개 시계분에서 threshold 이상 관측이 2회 이상. 2분 초과 정지 후 이력을 초기화합니다."

tool = wb[names[3]]
tool["A2"] = "24h XGBoost · 설비별 오경보"
for i, value in enumerate(["설비", "0.5 단독 사건", "0.5 단독 /12h", "0.36·2/5 사건",
                           "0.36·2/5 /12h", "허용치(건)", "판정"], 1):
    tool.cell(5, i).value = value
for row in range(6, 26):
    tool.cell(row, 7).value = "초과" if tool.cell(row, 5).value > 4 else "충족"
tool["A28"] = "/12h는 관측된 1분 행 기준 환산치입니다. 실제 12시간 근무조에서 발생한 횟수는 아닙니다."

regression = wb[names[4]]
regression["A2"] = "유형별 잔여시간 예측 · HGB-log와 LSTM"
for i, value in enumerate(["모델", "고장 유형", "학습 행", "검증 행", "MAE (시간)", "R²"], 1):
    regression.cell(5, i).value = value
regression["A14"] = "세 유형 모두 R²가 음수입니다. MAE 감소만으로 TTF 모델을 운영용으로 선정하지 않습니다."

wb.save(file)
print(file)
