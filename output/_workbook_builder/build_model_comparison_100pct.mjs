import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const dir = path.join(root, "output", "experiments", "model_comparison_100pct_12h_24h");
const data = JSON.parse(await fs.readFile(path.join(dir, "report.json"), "utf8"));
const connectedPath = path.join(root, "output", "experiments", "connected_lstm_100pct_12h_24h", "connected_metrics.csv");
const connectedRows = (await fs.readFile(connectedPath, "utf8")).replace(/^\uFEFF/, "").trim().split(/\r?\n/).map(line => line.split(","));
const connected = connectedRows.slice(1).map(row => Object.fromEntries(connectedRows[0].map((name, i) => [name, row[i]])));
const target = path.join(dir, "시계시간_12h_24h_100퍼센트_모델비교_연결검증포함.xlsx");
try { await fs.access(target); throw new Error(`Refusing to overwrite ${target}`); } catch (e) { if (e.code !== "ENOENT") throw e; }
const wb = Workbook.create();
const overview = wb.worksheets.add("검증 개요");
const classification = wb.worksheets.add("12h_24h_이진분류");
const regression = wb.worksheets.add("TTF_유형별_회귀");
const linked = wb.worksheets.add("분류_TTF_연결검증");
const limits = wb.worksheets.add("누수_한계_재현조건");
const navy = "#183153", dark = "#203040";
for (const sheet of [overview, classification, regression, linked, limits]) {
  sheet.showGridLines = false;
  sheet.getRange("A1:Q30").format.font = { name: "Arial", size: 10, color: dark };
  sheet.getRange("A:A").format.columnWidth = 30;
  sheet.getRange("B:Q").format.columnWidth = 20;
  sheet.getRange("A2").format.font = { name: "Arial", size: 15, bold: true, color: navy };
}
function header(sheet, range) {
  sheet.getRange(range).format.fill = navy;
  sheet.getRange(range).format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
  sheet.getRange(range).format.rowHeight = 32;
}
function put(sheet, row, values) { sheet.getRangeByIndexes(row - 1, 0, 1, values.length).values = [values]; }

put(overview, 2, ["실제 시계시간 12·24시간, 학습 100% 비교"]);
put(overview, 4, ["구분", "이번 실행의 상태"]); header(overview, "A4:B4");
put(overview, 5, ["데이터", "20개 설비 실제 train 센서·고장 로그"]);
put(overview, 6, ["입력 단위", "모든 4초 센서 행을 활용해 1분 특징으로 집계"]);
put(overview, 7, ["검증", "각 설비 시간 순 앞 80% 학습 / 뒤 20% 검증, 경계 전 24시간 제외"]);
put(overview, 8, ["분류 정답", "실제 시계시간 12/24시간 안에 저압·고압·누출 중 하나라도 발생"]);
put(overview, 9, ["분류 XGBoost", "기존 실제 데이터 100% 학습 모델의 저장 확률을 재평가해 동일 수치 확인"]);
put(overview, 10, ["분류 HGB", "실행 불가: Windows가 scikit-learn histogram 모듈을 차단"]);
put(overview, 11, ["회귀 LSTM", "유형별 독립 모델 3개를 전체 유효 학습 행으로 실행"]);
put(overview, 12, ["회귀 HGB-log", "실행 불가: 같은 scikit-learn 모듈 차단"]);
put(overview, 13, ["최종 판정", "HGB 실행 불가로 네 모델 전체 비교는 미완료. LSTM 최소 TTF 유형 선택도 낮은 정확도 확인"]);
put(overview, 15, ["주의", "20%는 검증 및 대시보드 재생에 사용 가능하나, 모델 선정에 쓰면 독립 최종 test는 아님"]);
overview.getRange("B:B").format.columnWidth = 105;

put(classification, 2, ["12·24시간 고장 위험 이진분류"]);
put(classification, 4, ["기간", "모델", "학습", "검증 행", "고장 행", "Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy", "AP / PR-AUC", "TN", "FP", "FN", "TP", "사건 탐지/전체", "10분 오경보"]);
header(classification, "A4:Q4");
for (let i = 0; i < data.classification.length; i++) {
  const r = data.classification[i];
  put(classification, 5 + i, [`${r.horizon_hours}시간`, r.model, r.train_fraction, r.validation_rows,
    r.positive_rows, r.accuracy, r.precision, r.recall, r.f1, r.balanced_accuracy,
    r.average_precision, r.tn, r.fp, r.fn, r.tp,
    `${r.detected_fault_events}/${r.logged_fault_events}`, r.false_alarm_events_10m]);
}
classification.getRange("F5:K6").setNumberFormat("0.00%");
classification.getRange("D5:E6").setNumberFormat("#,##0");
classification.getRange("L5:O6").setNumberFormat("#,##0");
classification.getRange("Q5:Q6").setNumberFormat("#,##0");
put(classification, 9, ["평가 가능 고장 사건", ...data.classification.map(r => `${r.horizon_hours}시간: ${r.detected_fault_events}/${r.evaluable_fault_events}`)]);
put(classification, 10, ["비교 기준", "24시간 XGBoost recovered baseline과 정확히 같은 확률·지표. 12시간은 동일 검증 행의 별도 모델."]);
put(classification, 11, ["HGB", "현재 PC 실행 정책으로 학습 불가. 모델 간 우열 판단 보류."]);
classification.getRange("B:Q").format.columnWidth = 20;
classification.freezePanes.freezeRows(4);

put(regression, 2, ["고장 유형별 독립 TTF 회귀"]);
put(regression, 4, ["모델", "고장 유형", "학습 행", "검증 행", "MAE(초)", "MAE(시간)", "RMSE(초)", "R²", "판정"]);
header(regression, "A4:I4");
for (let i = 0; i < data.regression.length; i++) {
  const r = data.regression[i];
  put(regression, 5 + i, [r.model, r.fault_type, Number(r.train_rows), Number(r.validation_rows),
    Number(r.mae_seconds), Number(r.mae_hours), Number(r.rmse_seconds), Number(r.r2),
    Number(r.r2) < 0 ? "잔여시간 예측으로 사용 부적합" : "추가 검증 필요"]);
}
regression.getRange("C5:G7").setNumberFormat("#,##0.00");
regression.getRange("H5:H7").setNumberFormat("0.0000");
put(regression, 10, ["HGB-log", "미실행", "Windows 실행 정책이 scikit-learn histogram 모듈을 차단했습니다."]);
put(regression, 11, ["학습 원칙", "분류가 위험으로 예측한 행만 사용하지 않았고, 유형별로 정답이 유효한 모든 학습 행을 사용했습니다."]);
put(regression, 12, ["평가 원칙", "회귀 성능은 분류 문턱값과 독립적으로 검증했습니다. 선택된 최소 TTF 유형의 정확도는 아직 검증되지 않았습니다."]);
regression.getRange("B:I").format.columnWidth = 25;
regression.getRange("I:I").format.columnWidth = 34;
regression.freezePanes.freezeRows(4);

put(linked, 2, ["분류 경보 뒤 최소 TTF로 고장 유형 선택"]);
put(linked, 4, ["기간", "고장 사건 전체", "분류 탐지 사건", "유형까지 정답", "탐지 중 유형 정확도", "전체 사건 중 유형 정답", "양성 경보 행", "행 유형 정확도", "선택 TTF MAE(시간)"]);
header(linked, "A4:I4");
connected.forEach((r, i) => put(linked, 5 + i, [`${r.horizon_hours}시간`, Number(r.logged_fault_events),
  Number(r.detected_fault_events), Number(r.type_correct_events),
  Number(r.type_accuracy_among_detected_events), Number(r.type_correct_over_all_fault_events),
  Number(r.alarm_positive_rows), Number(r.row_type_accuracy_on_detected_positives),
  Number(r.selected_min_ttf_mae_hours_on_detected_positives)]));
linked.getRange("E5:F6").setNumberFormat("0.00%");
linked.getRange("H5:H6").setNumberFormat("0.00%");
linked.getRange("I5:I6").setNumberFormat("#,##0.0");
put(linked, 9, ["판정", "현재 최솟값 선택으로 고장 유형을 신뢰성 있게 제시할 수 없습니다."]);
put(linked, 10, ["계산 방식", "실제 고장 12/24시간 전 경보 중 첫 경보에서 세 독립 LSTM 예측 TTF의 최솟값 유형을 선택했습니다."]);
linked.getRange("B:I").format.columnWidth = 24;
linked.getRange("B:B").format.columnWidth = 110;
linked.freezePanes.freezeRows(4);

put(limits, 2, ["데이터 흐름·누수 검사·해석 한계"]);
const notes = [
  ["원본", "20개 train 설비의 4초 센서 기록과 실제 fault log"],
  ["1분 집계", "1분 안의 현재 센서 통계와 과거 5/30분 rolling·EWMA·변화율. 긴 정지 후 이력 초기화"],
  ["금지 입력", "원본 TTF, 실제 고장시각, 미래 유형, 절대 시각, Lot, runnum"],
  ["분류 라벨", "각 행 이후 H=12/24시간 실제 시계시간 안에 세 유형 중 하나라도 고장 발생 시 1"],
  ["검열", "H만큼 미래를 관찰하지 못한 정상 후보는 평가에서 제외"],
  ["분할", "설비별 시간 순 80:20; 학습 끝 24시간 purge; 같은 고장 사건이 양쪽 정답에 섞이지 않도록 확인"],
  ["TTF 회귀", "세 고장 유형의 다음 실제 fault까지 초 단위. 해당 고장이 학습 cutoff 이후면 학습 정답에서 제외"],
  ["모델 연결", "분류 위험 시 세 유형 TTF를 각각 예측하고 최소값을 선택하는 것은 배포 로직; 이번 회귀 지표는 독립 평가"],
  ["HGB 차단", "WinError 4551, sklearn histogram.cp312-win_amd64.pyd. HGB 수치는 이번 보고서에 없음"],
  ["비교 한계", "12h와 24h 검증 행은 같지만 고장 비율이 달라 AP를 수치만으로 직접 우열 판단하지 않음"],
  ["최종 test", "별도 후속 test 구간 없음. 검증 20%를 대시보드 재생에 쓰면 데모이지 독립 최종 성능은 아님"],
];
put(limits, 4, ["항목", "확인 내용"]); header(limits, "A4:B4");
notes.forEach((r, i) => put(limits, 5 + i, r));
limits.getRange("B:B").format.columnWidth = 110;

wb.recalculate();
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(target);
console.log(`saved ${target}`);
