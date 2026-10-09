import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.resolve(process.cwd());
const outDir = path.join(root, "output", "experiments", "horizon_comparison_12h_24h");
const report = JSON.parse(await fs.readFile(path.join(outDir, "comparison_report.json"), "utf8"));
const prep = JSON.parse(await fs.readFile(path.join(root, "output", "baseline_recovery", "preparation_report.json"), "utf8"));
const [h12, h24] = report.results;
console.log("builder: loaded results");
if (h12.horizon_hours !== 12 || h24.horizon_hours !== 24) throw new Error("Unexpected horizon order");
const trainRows24 = prep.tools.reduce((sum, item) => sum + item.train_rows, 0);
const trainPos24 = prep.tools.reduce((sum, item) => sum + item.train_positive, 0);

const wb = Workbook.create();
const summary = wb.worksheets.add("요약");
const classification = wb.worksheets.add("이진분류");
const events = wb.worksheets.add("사건탐지");
const method = wb.worksheets.add("검증조건");
const sheets = [summary, classification, events, method];
console.log("builder: sheets created");
const navy = "#183153", blue = "#285E8E", pale = "#E8F0F7", ink = "#1F2937", muted = "#64748B";
for (const s of sheets) {
  s.showGridLines = false;
  s.getRange("A1:R42").format.font = { name: "Arial", size: 10, color: ink };
  s.getRange("A:A").format.columnWidth = 27;
  s.getRange("B:Q").format.columnWidth = 16;
  s.getRange("A2").format.font = { name: "Arial", size: 15, bold: true, color: navy };
  s.getRange("A2:Q2").format.rowHeight = 28;
}
console.log("builder: initial formatting complete");

function header(s, range) {
  s.getRange(range).format.fill = navy;
  s.getRange(range).format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
  s.getRange(range).format.rowHeight = 27;
}
function section(s, range) {
  s.getRange(range).format.fill = pale;
  s.getRange(range).format.font = { name: "Arial", size: 10, bold: true, color: blue };
  s.getRange(range).format.rowHeight = 23;
}
function pct(s, range) { s.getRange(range).setNumberFormat("0.00%"); }
function count(s, range) { s.getRange(range).setNumberFormat("#,##0"); }
function dec(s, range) { s.getRange(range).setNumberFormat("#,##0.00"); }
function writeRows(s, startRow, rows) {
  s.getRangeByIndexes(startRow - 1, 0, rows.length, rows[0].length).values = rows;
}

summary.getRange("A2").values = [["12시간·24시간 고장예측 검증 비교"]];
summary.getRange("A4:B8").values = [
  ["원본 센서 데이터", "20개 설비의 실제 train 데이터"],
  ["검증 행", h12.validation_rows],
  ["검증 고장 사건", h12.logged_fault_events],
  ["분류 임계값", 0.5],
  ["검증 구간", "설비별 시간순 마지막 20%"],
];
summary.getRange("B:B").format.columnWidth = 30;
summary.getRange("C:C").format.columnWidth = 18;
summary.getRange("D:D").format.columnWidth = 22;
summary.getRange("A10:D10").values = [["지표", "12시간", "24시간", "12시간 − 24시간"]];
header(summary, "A10:D10");
const summaryRows = [
  ["AP / PR-AUC", h12.average_precision, h24.average_precision],
  ["Precision (행)", h12.precision, h24.precision],
  ["Recall (행)", h12.recall, h24.recall],
  ["F1 (행)", h12.f1, h24.f1],
  ["Balanced Accuracy", h12.balanced_accuracy, h24.balanced_accuracy],
  ["Accuracy", h12.accuracy, h24.accuracy],
  ["고장 사건 Recall (전체)", h12.all_fault_event_recall, h24.all_fault_event_recall],
  ["고장 사건 Recall (평가 가능)", h12.evaluable_fault_event_recall, h24.evaluable_fault_event_recall],
  ["탐지 고장 사건 수", h12.detected_fault_events, h24.detected_fault_events],
  ["10분 기준 오경보 사건", h12.false_alarm_events_10m, h24.false_alarm_events_10m],
  ["설비 관측일당 오경보", h12.false_alarm_events_per_observed_tool_day, h24.false_alarm_events_per_observed_tool_day],
  ["최초 경보 lead 중앙값 (시간)", h12.first_alert_lead_median_minutes / 60, h24.first_alert_lead_median_minutes / 60],
];
writeRows(summary, 11, summaryRows.map(([name, a, b]) => [name, a, b, a - b]));
pct(summary, "B11:D18"); count(summary, "B19:D20"); dec(summary, "B21:D22");
section(summary, "A16:D16");
summary.getRange("A24").values = [["해석"]]; section(summary, "A24:D24");
summary.getRange("A25").values = [["12시간과 24시간은 동일한 검증 시점을 평가했지만 양성 기준이 달라 AP·Precision을 단순 우열로 해석하면 안 됩니다."]];
summary.getRange("A26").values = [["사건 탐지와 오경보를 함께 비교하세요. 이 결과는 최종 미사용 test가 아니라 기존 검증 구간의 재평가입니다."]];
summary.getRange("A25:D26").format.font = { name: "Arial", size: 10, color: muted };
summary.getRange("A:A").format.columnWidth = 31;
console.log("builder: summary complete");

classification.getRange("A2").values = [["이진분류 성능 비교"]];
classification.getRange("A4:Q4").values = [["기간", "모델", "학습 행", "학습 양성", "검증 행", "검증 양성", "검증 음성", "Accuracy", "Precision", "Recall", "F1", "Balanced Acc.", "AP", "TN", "FP", "FN", "TP"]];
header(classification, "A4:Q4");
writeRows(classification, 5, [[
  "12시간", "XGBoost", h12.train_rows, h12.train_positive_rows,
  h12.validation_rows, h12.positive_rows, h12.negative_rows,
  h12.accuracy, h12.precision, h12.recall, h12.f1, h12.balanced_accuracy,
  h12.average_precision, h12.tn, h12.fp, h12.fn, h12.tp,
], [
  "24시간", "XGBoost", trainRows24, trainPos24,
  h24.validation_rows, h24.positive_rows, h24.negative_rows,
  h24.accuracy, h24.precision, h24.recall, h24.f1, h24.balanced_accuracy,
  h24.average_precision, h24.tn, h24.fp, h24.fn, h24.tp,
]]);
count(classification, "C5:G6"); pct(classification, "H5:M6"); count(classification, "N5:Q6");
classification.getRange("A9").values = [["판정 기준: 해당 시점 이후 지정 기간 안에 pressure_low / pressure_high / leak 중 하나가 발생하면 양성(1)."]];
classification.getRange("A10").values = [["두 기간 모두 임계값 0.5와 동일한 85개 입력 변수·분할을 사용했습니다. 12시간 모델만 새로 학습했습니다."]];
classification.getRange("C:Q").format.columnWidth = 15;
classification.getRange("L:M").format.columnWidth = 18;
classification.freezePanes.freezeRows(4);
console.log("builder: classification complete");

events.getRange("A2").values = [["고장 사건 탐지·오경보 비교"]];
events.getRange("A4:L4").values = [["기간", "전체 사건", "평가 가능", "탐지", "전체 Recall", "평가 가능 Recall", "놓친 사건", "오경보(10분)", "설비 관측일당", "lead 중앙(분)", "lead 최소(분)", "lead 최대(분)"]];
header(events, "A4:L4");
writeRows(events, 5, [h12, h24].map(r => [
  `${r.horizon_hours}시간`, r.logged_fault_events, r.evaluable_fault_events,
  r.detected_fault_events, r.all_fault_event_recall, r.evaluable_fault_event_recall,
  r.logged_fault_events - r.detected_fault_events, r.false_alarm_events_10m,
  r.false_alarm_events_per_observed_tool_day, r.first_alert_lead_median_minutes,
  r.first_alert_lead_min_minutes, r.first_alert_lead_max_minutes,
]));
count(events, "B5:D6"); pct(events, "E5:F6"); count(events, "G5:H6"); dec(events, "I5:L6");
events.getRange("A9").values = [["탐지: 실제 고장 전 H시간 이내에 해당 설비에서 경보가 1회 이상 발생. 고장 시각의 행은 제외."]];
events.getRange("A10").values = [["평가 가능: 고장 전 H시간 안에 검증 센서 행이 1개 이상 있는 사건. 전체 분모에는 센서 행이 없는 사건도 포함."]];
events.getRange("A11").values = [["오경보: 정상 라벨의 경보를 설비별 10분 이내 연속 건으로 병합. 관측일은 1분 단위 검증 행 / 1,440."]];
events.getRange("B:L").format.columnWidth = 18;
events.getRange("E:F").format.columnWidth = 21;
events.freezePanes.freezeRows(4);
console.log("builder: events complete");

method.getRange("A2").values = [["검증 조건과 재현 범위"]];
method.getRange("A4:B4").values = [["항목", "설명"]]; header(method, "A4:B4");
writeRows(method, 5, [
  ["실제 데이터", "20개 설비 train 센서를 1분 단위로 집계한 기존 prepared parquet 사용"],
  ["시간 분할", "설비별 과거 80% 학습 / 이후 20% 검증, 경계 전 24시간 제외"],
  ["공통 검증 모집단", "두 모델 모두 같은 1,142,112개 검증 행; 24시간 기준 완전 라벨만 사용"],
  ["라벨", "다음 실제 고장까지 12시간 또는 24시간 이내면 1; TTF·고장시각은 입력 변수 아님"],
  ["모델", "24시간 Recovered Baseline 고정; 12시간만 같은 XGBoost 설정으로 재학습"],
  ["입력 특징", "기준 모델의 85개 현재/과거 센서·공정·시간 변화 특징 사용"],
  ["임계값", "양쪽 모두 0.5 고정; 검증 구간에서 재조정하지 않음"],
  ["사건 총수", "검증 구간 fault log의 고유 (설비, 고장 유형, 시각) 181건"],
  ["AP 해석", "12시간과 24시간은 양성 비율이 달라 AP 수치를 직접 비교하기 어려움"],
  ["TTF 회귀", "이번 재실험에서 수행하지 않음. MAE/RMSE/R²를 임의로 기입하지 않음"],
  ["독립 시험", "공식 test 및 새로운 시간상 후속 구간은 사용하지 않음. 최종 운영 성능 주장 불가"],
  ["참조 보고서", "LS_team6_3h_6h_검증보고서.xlsx: 시트 구성 참고, 3h/6h 성능 수치는 사용하지 않음"],
]);
method.getRange("B:B").format.columnWidth = 95;
method.getRange("A:A").format.columnWidth = 25;
method.freezePanes.freezeRows(4);
console.log("builder: methods complete");

wb.recalculate();
console.log("builder: recalculated");
const check = await wb.inspect({ kind: "region", sheetId: "요약", range: "A10:D22", maxChars: 4500 });
console.log(check.ndjson);
const output = await SpreadsheetFile.exportXlsx(wb);
const workbookPath = path.join(outDir, "LS_team6_12h_24h_검증보고서.xlsx");
await output.save(workbookPath);
console.log(`saved ${workbookPath}`);
