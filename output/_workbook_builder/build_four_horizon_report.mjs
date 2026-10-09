import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const dir = path.join(root, "output", "experiments", "horizon_learning_curve_3h_6h_12h_24h");
const report = JSON.parse(await fs.readFile(path.join(dir, "comparison_report.json"), "utf8"));
const old = JSON.parse(await fs.readFile(path.join(dir, "historical_3h_6h_reference.json"), "utf8"));
const rows = report.results;
if (rows.length !== 8) throw new Error("Expected 8 new model results");
const wb = Workbook.create();
const overview = wb.worksheets.add("요약");
const classification = wb.worksheets.add("이진분류");
const events = wb.worksheets.add("사건탐지");
const historical = wb.worksheets.add("이전3h6h참고");
const method = wb.worksheets.add("검증조건");
const ink = "#203040", navy = "#183153", blue = "#235F8C", pale = "#E7F0F8";

for (const s of [overview, classification, events, historical, method]) {
  s.showGridLines = false;
  s.getRange("A1:R35").format.font = { name: "Arial", size: 10, color: ink };
  s.getRange("A2").format.font = { name: "Arial", size: 15, bold: true, color: navy };
  s.getRange("A:A").format.columnWidth = 24;
  s.getRange("B:R").format.columnWidth = 16;
}
const title = (s, text) => s.getRange("A2").values = [[text]];
function header(s, range) {
  s.getRange(range).format.fill = navy;
  s.getRange(range).format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
  s.getRange(range).format.rowHeight = 26;
}
function section(s, range) {
  s.getRange(range).format.fill = pale;
  s.getRange(range).format.font = { name: "Arial", size: 10, bold: true, color: blue };
}
function writeRows(s, start, values) {
  s.getRangeByIndexes(start - 1, 0, values.length, values[0].length).values = values;
}

title(overview, "3·6·12·24시간 고장예측 비교 — 학습 1% / 100%");
overview.getRange("A4:B7").values = [
  ["데이터", "20개 실제 설비 train"],
  ["검증 행", rows[0].validation_rows],
  ["고장 사건", rows[0].logged_fault_events],
  ["예측 임계값", 0.5],
];
overview.getRange("A9:L9").values = [["예측기간", "학습비율", "Precision", "행 Recall", "F1", "AP", "Balanced Acc.", "사건 탐지", "사건 Recall", "오경보(10분)", "오경보/관측일", "평가 가능 사건"]];
header(overview, "A9:L9");
writeRows(overview, 10, rows.map(r => [
  `${r.horizon_hours}시간`, r.train_fraction, r.precision, r.recall, r.f1,
  r.average_precision, r.balanced_accuracy, r.detected_fault_events,
  r.all_fault_event_recall, r.false_alarm_events_10m,
  r.false_alarm_events_per_observed_tool_day, r.evaluable_fault_events,
]));
overview.getRange("C10:G17").setNumberFormat("0.00%");
overview.getRange("I10:I17").setNumberFormat("0.00%");
overview.getRange("H10:H17").setNumberFormat("#,##0");
overview.getRange("J10:J17").setNumberFormat("#,##0");
overview.getRange("K10:K17").setNumberFormat("#,##0.00");
overview.getRange("L10:L17").setNumberFormat("#,##0");
overview.getRange("A20").values = [["읽는 방법"]]; section(overview, "A20:L20");
overview.getRange("A21").values = [["행 Recall·F1은 매분 예측 기준, 사건 Recall은 실제 고장 181건 중 한 번이라도 미리 경보한 비율입니다."]];
overview.getRange("A22").values = [["예전 3·6시간 파일은 기간별 검증 행을 달리 골랐습니다. 아래 새 수치와 옛 수치를 직접 합치거나 순위를 매기면 안 됩니다."]];
overview.getRange("A23").values = [["이번 결과는 네 기간에 같은 검증 행을 사용한 새 XGBoost 실험이며, 삭제된 원본 12·24시간 성적의 복원값이 아닙니다."]];
overview.getRange("A:A").format.columnWidth = 27;
overview.getRange("B:B").format.columnWidth = 20;
overview.getRange("C:G").format.columnWidth = 17;
overview.getRange("H:L").format.columnWidth = 19;

title(classification, "이진분류: 학습량과 예측기간별 행 단위 성능");
classification.getRange("A4:Q4").values = [["기간", "학습비율", "학습 행", "학습 양성", "검증 행", "검증 양성", "검증 음성", "Accuracy", "Precision", "Recall", "F1", "Balanced Acc.", "AP", "TN", "FP", "FN", "TP"]];
header(classification, "A4:Q4");
writeRows(classification, 5, rows.map(r => [
  `${r.horizon_hours}시간`, r.train_fraction, r.train_rows, r.train_positive_rows,
  r.validation_rows, r.positive_rows, r.negative_rows, r.accuracy,
  r.precision, r.recall, r.f1, r.balanced_accuracy, r.average_precision,
  r.tn, r.fp, r.fn, r.tp,
]));
classification.getRange("C5:G12").setNumberFormat("#,##0");
classification.getRange("H5:M12").setNumberFormat("0.00%");
classification.getRange("N5:Q12").setNumberFormat("#,##0");
classification.getRange("A15").values = [["양성: 해당 시점 이후 지정 기간 안에 세 고장 유형 중 하나가 실제로 발생. 검증 행은 모든 기간에 동일합니다."]];
classification.getRange("C:Q").format.columnWidth = 16;
classification.freezePanes.freezeRows(4);

title(events, "실제 고장 사건 탐지와 10분 기준 오경보");
events.getRange("A4:K4").values = [["기간", "학습비율", "전체 사건", "평가 가능", "탐지", "전체 Recall", "평가 가능 Recall", "놓친 사건", "오경보 사건", "오경보/관측일", "최초 경보 lead 중앙(분)"]];
header(events, "A4:K4");
writeRows(events, 5, rows.map(r => [
  `${r.horizon_hours}시간`, r.train_fraction, r.logged_fault_events,
  r.evaluable_fault_events, r.detected_fault_events, r.all_fault_event_recall,
  r.evaluable_fault_event_recall, r.logged_fault_events - r.detected_fault_events,
  r.false_alarm_events_10m, r.false_alarm_events_per_observed_tool_day,
  r.first_alert_lead_median_minutes,
]));
events.getRange("C5:E12").setNumberFormat("#,##0");
events.getRange("F5:G12").setNumberFormat("0.00%");
events.getRange("H5:I12").setNumberFormat("#,##0");
events.getRange("J5:K12").setNumberFormat("#,##0.00");
events.getRange("A15").values = [["전체 사건 181건 중 고장 전 해당 기간에 검증 센서 행이 있는 사건만 '평가 가능'에 포함됩니다."]];
events.getRange("A16").values = [["오경보 사건은 정상 행 경보를 설비별로 10분 이내 연속 묶음으로 합친 수입니다."]];
events.getRange("F:K").format.columnWidth = 22;
events.freezePanes.freezeRows(4);

title(historical, "이전 3·6시간 파일의 수치 — 비교 참고용");
historical.getRange("A4:N4").values = [["기간", "모델", "학습비율", "검증 행", "검증 양성", "Precision", "행 Recall", "F1", "AP", "사건 탐지", "사건 Recall", "TN", "FP", "출처"]];
header(historical, "A4:N4");
writeRows(historical, 5, old.rows.map(r => [
  r[0], r[1], r[2], r[4], r[6], r[7], r[8], r[9], r[11],
  r[16], typeof r[16] === "string" ? Number(r[16].split("/")[0]) / Number(r[16].split("/")[1]) : null,
  r[12], r[13], "기존 3h/6h 보고서",
]));
historical.getRange("F5:I10").setNumberFormat("0.00%");
historical.getRange("K5:K10").setNumberFormat("0.00%");
historical.getRange("D5:E10").setNumberFormat("#,##0");
historical.getRange("L5:M10").setNumberFormat("#,##0");
historical.getRange("A13").values = [["예: 기존 6시간 HGB 100%의 75/80=93.75%는 고장 사건 Recall입니다. 같은 행의 행 Recall은 약 76.05%, F1은 약 62.29%입니다."]];
historical.getRange("A14").values = [["기존 3시간 XGBoost의 검증 행은 38,435개지만 새 실험은 모든 기간에 1,142,112개입니다. 옛 점수와 새 점수를 직접 비교하지 마세요."]];
historical.getRange("B:N").format.columnWidth = 18;
historical.getRange("N:N").format.columnWidth = 24;

title(method, "검증 조건 및 원본 보고서와 다른 점");
method.getRange("A4:B4").values = [["항목", "내용"]]; header(method, "A4:B4");
writeRows(method, 5, [
  ["기간", "3·6·12·24시간, 세 유형 중 가장 가까운 실제 고장으로 라벨 생성"],
  ["입력", "기준 모델의 85개 현재·과거 센서 및 공정 특징; TTF·미래 고장 시각 제외"],
  ["학습 1%", "설비별 과거 학습 행 전체에서 seed=42로 같은 1% 인덱스를 무작위 추출"],
  ["학습 100%", "설비별 과거 학습 행 전체 사용"],
  ["모델", "XGBoost. 12·24시간 100%는 기존 저장 모델·확률 그대로 재사용"],
  ["HGB", "현재 PC의 scikit-learn HGB DLL 로드 오류로 새 HGB 성적은 생성하지 않음"],
  ["검증", "20개 설비별 시간순 마지막 20%; 경계 전 24시간 purge; 1,142,112행 공통"],
  ["임계값", "전 모델 0.5 고정; validation에서 재선정하지 않음"],
  ["사건 탐지", "고장 전 H시간에 1회 이상 경보; 전체 181건, 평가 가능 수는 기간별로 다름"],
  ["오경보", "정상 행 경보를 동일 설비에서 10분 이내 이어지면 1사건으로 병합"],
  ["TTF 회귀", "이 재실험에는 포함하지 않음; 과거 3·6시간 값도 새 모델 결과로 옮기지 않음"],
  ["중요한 차이", "과거 3·6시간 파일은 검증 모집단이 기간마다 다르며, 삭제된 12·24시간 원본 수치는 찾지 못함"],
  ["검증 한계", "모든 값은 기존 시간상 검증 구간 결과이며 독립된 최종 test 점수가 아님"],
]);
method.getRange("B:B").format.columnWidth = 100;
method.getRange("A:A").format.columnWidth = 22;
method.freezePanes.freezeRows(4);

wb.recalculate();
const check = await wb.inspect({ kind: "region", sheetId: "요약", range: "A9:L17", maxChars: 2000 });
console.log(check.ndjson.slice(0, 2000));
const target = path.join(dir, "시계시간_3h_6h_12h_24h_학습비율_검증보고서.xlsx");
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(target);
console.log(`saved ${target}`);
