import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const dir = path.join(root, "output", "experiments", "historical_report_recovery");
const archive = JSON.parse(await fs.readFile(path.join(dir, "archived_results.json"), "utf8"));
const wb = Workbook.create();
const guide = wb.worksheets.add("안내");
const continuous = wb.worksheets.add("연속가동_3_6_12_24");
const calendar = wb.worksheets.add("시계시간_12_24_분류");
const regression = wb.worksheets.add("시계시간_TTF회귀");
const joint = wb.worksheets.add("시계시간_연결결과");
const ink = "#203040", navy = "#183153", muted = "#62748A";
for (const sheet of [guide, continuous, calendar, regression, joint]) {
  sheet.showGridLines = false;
  sheet.getRange("A1:Q28").format.font = { name: "Arial", size: 10, color: ink };
  sheet.getRange("A2").format.font = { name: "Arial", size: 15, bold: true, color: navy };
  sheet.getRange("A:A").format.columnWidth = 25;
  sheet.getRange("B:Q").format.columnWidth = 17;
}
function header(sheet, range) {
  sheet.getRange(range).format.fill = navy;
  sheet.getRange(range).format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
  sheet.getRange(range).format.rowHeight = 31;
}
function rows(sheet, start, matrix) {
  sheet.getRangeByIndexes(start - 1, 0, matrix.length, matrix[0].length).values = matrix;
}

guide.getRange("A2").values = [["삭제된 검증보고서의 대화 기록 복원본"]];
guide.getRange("A4:B10").values = [
  ["상태", "원본 엑셀 파일은 현재 저장 경로·자동복구·휴지통에서 찾지 못했습니다."],
  ["복원 근거", "이전 Codex 대화에 남은 실행 결과의 숫자와 원래 저장 경로"],
  ["보고서 ①", "연속 가동시간 기준 3·6·12·24시간 — 90%대 탐지율이 나온 실험"],
  ["보고서 ②", "실제 시계시간 기준 12·24시간 — 설비 정지 시간도 예측 기간에 포함"],
  ["주의", "두 실험은 시간 정의와 평가 가능한 행·사건 수가 달라 직접 비교할 수 없습니다."],
  ["한계", "원래 7개 시트 전체, 사건별 2,172행 원장, 누락된 지표는 복원하지 못했습니다."],
  ["TTF", "복원된 회귀 수치는 R²가 모두 음수입니다. 정확한 잔여시간 예측으로 해석하지 마세요."],
];
guide.getRange("A12").values = [["원래 저장 경로"]];
guide.getRange("A13:B14").values = [
  ["연속가동", archive.missing_original_filenames[0]],
  ["시계시간", archive.missing_original_filenames[1]],
];
guide.getRange("A16").values = [["출처: 이전 대화 기록의 실행 결과. 이 파일은 원본 엑셀의 복사본이나 전체 재현본이 아니라 확인 가능한 숫자의 복원 요약입니다."]];
guide.getRange("B:B").format.columnWidth = 107;
guide.getRange("A16").format.font = { name: "Arial", size: 10, color: muted };

continuous.getRange("A2").values = [["연속 가동시간 기준: 3·6·12·24시간"]];
continuous.getRange("A4:L4").values = [["기간", "검증 행", "정상 행", "고장 행", "검열 행", "고장 비율", "AP", "탐지 사건", "사건 Recall", "첫 경보 중앙(분)", "1시간 이상 선행", "H시간 이상 선행"]];
header(continuous, "A4:L4");
rows(continuous, 5, archive.continuous_operating_time_summary.map(r => [
  `${r.horizon_hours}시간`, r.validation_rows, r.normal_rows, r.positive_rows,
  r.censored_rows, r.positive_prevalence, r.average_precision,
  `${r.detected_fault_events}/${r.fault_event_denominator}`,
  r.fault_event_recall, r.first_alert_lead_median_minutes,
  r.detected_at_least_1h_before_fault,
  r.detected_at_least_h_hours_before_fault,
]));
continuous.getRange("B5:E8").setNumberFormat("#,##0");
continuous.getRange("F5:G8").setNumberFormat("0.00%");
continuous.getRange("I5:I8").setNumberFormat("0.00%");
continuous.getRange("J5:J8").setNumberFormat("#,##0.0");
continuous.getRange("A11").values = [["이전 보고서의 12시간 79/80건(98.75%), 24시간 75/80건(93.75%)이 기억하신 90%대 사건 탐지율입니다."]];
continuous.getRange("A12").values = [["그러나 최초 경보 중앙값은 약 28분 전이고, 12·24시간 이상 미리 경보한 사건은 0건이었습니다."]];
continuous.getRange("A13").values = [["24시간 AP 91.75%는 평가 행 중 고장 비율 61.51%인 제한된 모집단에서 측정됐습니다."]];
continuous.getRange("B:L").format.columnWidth = 20;
continuous.freezePanes.freezeRows(4);

calendar.getRange("A2").values = [["실제 시계시간 기준: 12·24시간 분류"]];
calendar.getRange("A4:P4").values = [["기간", "모델", "학습비율", "검증 행", "고장 행", "Precision", "행 Recall", "F1*", "AP", "탐지 사건", "평가 가능", "사건 Recall", "센서 없음", "첫 경보 중앙(분)", "오경보/관측시간", "전체 고장"]];
header(calendar, "A4:P4");
rows(calendar, 5, archive.calendar_time_classification.map(r => [
  `${r.horizon_hours}시간`, r.model, r.train_fraction,
  r.validation_rows, r.positive_rows, r.precision, r.recall,
  r.f1_derived_from_precision_recall, r.average_precision,
  r.detected_fault_events, r.evaluable_fault_events,
  r.event_recall_evaluable, r.events_without_prior_sensor,
  r.first_alert_lead_median_minutes, r.false_alerts_per_observed_hour,
  r.all_logged_fault_events,
]));
calendar.getRange("D5:E10").setNumberFormat("#,##0");
calendar.getRange("F5:I10").setNumberFormat("0.00%");
calendar.getRange("J5:K10").setNumberFormat("#,##0");
calendar.getRange("L5:L10").setNumberFormat("0.00%");
calendar.getRange("M5:M10").setNumberFormat("#,##0");
calendar.getRange("N5:O10").setNumberFormat("#,##0.00");
calendar.getRange("A13").values = [["* F1은 기록에 남은 Precision과 Recall로 재계산했습니다. 원본 워크북에서 직접 복사한 값은 아닙니다."]];
calendar.getRange("A14").values = [["시계시간 XGBoost 100% 사건 탐지는 12시간 105/154건, 24시간 111/163건으로 연속가동 보고서의 90%대와 다릅니다."]];
calendar.getRange("D:P").format.columnWidth = 19;
calendar.freezePanes.freezeRows(4);

regression.getRange("A2").values = [["실제 시계시간 기준: 유형별 TTF 회귀"]];
regression.getRange("A4:G4").values = [["고장 유형", "모델", "학습비율", "검증 행", "사건", "MAE(시간)", "R²"]];
header(regression, "A4:G4");
rows(regression, 5, archive.calendar_time_regression_partial.map(r => {
  const key = r.model_fault_key;
  const fault = key.endsWith("pressure_low") ? "Pressure Low" : key.endsWith("pressure_high") ? "Pressure High" : "Leak";
  const model = key.startsWith("lstm") ? "LSTM" : "HGB-log";
  const fraction = key.includes("1pct") ? "1%" : "100%";
  return [fault, model, fraction, r.rows, r.events, r.mae_seconds/3600, r.r2];
}));
regression.getRange("D5:E13").setNumberFormat("#,##0");
regression.getRange("F5:G13").setNumberFormat("#,##0.00");
regression.getRange("A16").values = [["기록에서 확인된 MAE와 R²만 표시합니다. 원본의 RMSE 등 확인되지 않은 칸은 추정하지 않았습니다."]];
regression.getRange("B:G").format.columnWidth = 20;

joint.getRange("A2").values = [["실제 시계시간 기준: 분류와 TTF 연결 결과"]];
joint.getRange("A4:J4").values = [["기간", "TTF 모델", "학습비율", "평가 가능 사건", "탐지 사건", "유형 정답 사건", "유형 정답/평가 가능", "정답 유형 RUL MAE(시간)", "Low 선택 행", "Leak 선택 행"]];
header(joint, "A4:J4");
rows(joint, 5, archive.calendar_time_joint_partial.map(r => {
  const key = r.system_key;
  const horizon = Number(key.match(/h(\d+)/)[1])/3600;
  const model = key.includes("lstm") ? "LSTM" : "HGB-log";
  const fraction = key.includes("1pct") ? "1%" : "100%";
  return [`${horizon}시간`, model, fraction, r.events, r.detected_events,
          r.type_correct_events, r.type_correct_event_recall,
          r.rul_mae_seconds_when_gate_and_type_correct/3600,
          r.selected_type_counts_on_alerts.pressure_low,
          r.selected_type_counts_on_alerts.leak];
}));
joint.getRange("D5:F10").setNumberFormat("#,##0");
joint.getRange("G5:G10").setNumberFormat("0.00%");
joint.getRange("H5:H10").setNumberFormat("#,##0.0");
joint.getRange("I5:J10").setNumberFormat("#,##0");
joint.getRange("B:J").format.columnWidth = 23;

wb.recalculate();
const check = await wb.inspect({ kind: "region", sheetId: "연속가동_3_6_12_24", range: "A4:L8", maxChars: 2200 });
console.log(check.ndjson.slice(0, 2200));
const target = path.join(dir, "대화기록_복원_12h_24h_검증요약.xlsx");
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(target);
console.log(`saved ${target}`);
