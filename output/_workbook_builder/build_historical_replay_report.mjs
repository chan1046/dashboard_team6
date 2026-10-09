import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const dir = path.join(root, "output", "experiments", "historical_exact_replay", "reproduced");
const replay = JSON.parse(await fs.readFile(path.join(dir, "xgb_replay_report.json"), "utf8"));
if (replay.results.length !== 4 || replay.results.some(r => !r.matches_archived)) {
  throw new Error("Four exact matches against archived outputs are required");
}
const preparation = JSON.parse(await fs.readFile(path.join(dir, "preparation_report.json"), "utf8"));
const faultPopulation = JSON.parse(await fs.readFile(path.join(dir, "validation_fault_population.json"), "utf8"));
const calendar = JSON.parse(await fs.readFile(path.join(root, "output", "experiments", "horizon_comparison_12h_24h", "comparison_report.json"), "utf8"));
const target = path.join(dir, "과거_XGBoost100_3h_6h_12h_24h_재실행_검증보고서.xlsx");
try { await fs.access(target); throw new Error(`Refusing to overwrite ${target}`); } catch (e) { if (e.code !== "ENOENT") throw e; }
const wb = Workbook.create();
const headline = wb.worksheets.add("재현 결과");
const comparison = wb.worksheets.add("현재 기준과 비교");
const definition = wb.worksheets.add("정답과 평가 대상");
const ink = "#203040", navy = "#183153", red = "#A62424";
for (const sheet of [headline, comparison, definition]) {
  sheet.showGridLines = false;
  sheet.getRange("A1:U35").format.font = { name: "Arial", size: 10, color: ink };
  sheet.getRange("A2").format.font = { name: "Arial", size: 15, bold: true, color: navy };
  sheet.getRange("A:A").format.columnWidth = 26;
  sheet.getRange("B:U").format.columnWidth = 19;
}
function put(sheet, row, items) { sheet.getRangeByIndexes(row - 1, 0, 1, items.length).values = [items]; }
function header(sheet, range) {
  sheet.getRange(range).format.fill = navy;
  sheet.getRange(range).format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
  sheet.getRange(range).format.rowHeight = 32;
}

put(headline, 2, ["기억하신 90%대 결과: 실제 train 데이터 재실행"]);
put(headline, 4, ["예측기간", "학습 행", "검증 행", "고장 행", "정상 행", "제외 행", "Precision", "Recall", "F1", "Balanced Accuracy", "AP", "TN", "FP", "FN", "TP", "고장 사건 탐지", "첫 경보 중앙(분)", "10분 오경보", "기록 일치", "전체 기록 고장 대비"]);
header(headline, "A4:T4");
for (let i = 0; i < replay.results.length; i++) {
  const r = replay.results[i];
  const censored = Object.values(preparation.tools).reduce((sum, t) => sum + t.horizons[String(r.horizon_seconds)].validation.censored, 0);
  put(headline, 5 + i, [`${r.horizon_hours}시간`, r.train_rows, r.validation_rows, r.positive_rows,
    r.validation_rows - r.positive_rows, censored, r.precision, r.recall, r.f1,
    r.balanced_accuracy, r.average_precision, r.tn, r.fp, r.fn, r.tp,
    `${r.detected_events}/${r.events}`, r.median_lead_minutes,
    r.false_alert_episodes_10m_cooldown, r.matches_archived ? "일치" : "불일치",
    r.detected_events / faultPopulation.all_unique_logged_validation_fault_events]);
}
headline.getRange("B5:F8").setNumberFormat("#,##0");
headline.getRange("G5:K8").setNumberFormat("0.00%");
headline.getRange("L5:O8").setNumberFormat("#,##0");
headline.getRange("Q5:Q8").setNumberFormat("0.0");
headline.getRange("R5:R8").setNumberFormat("#,##0");
headline.getRange("T5:T8").setNumberFormat("0.00%");
put(headline, 11, ["반드시 확인", "12시간·24시간 Recall과 F1이 높았던 것은 사실입니다. 평가 가능한 행이 크게 줄어든 조건부 결과입니다."]);
put(headline, 12, ["미리 경보", "12시간/24시간 고장 사건 중 해당 기간만큼 일찍 경보한 사건은 0건. 최초 경보 중앙값은 약 28분 전입니다."]);
headline.getRange("A11:B12").format.font = { name: "Arial", size: 10, bold: true, color: red };
headline.getRange("B:B").format.columnWidth = 23;
headline.getRange("F:F").format.columnWidth = 20;
headline.getRange("P:P").format.columnWidth = 20;
headline.freezePanes.freezeRows(4);

put(comparison, 2, ["동일한 12·24시간 이름이지만 평가 조건이 다른 두 실험"]);
put(comparison, 4, ["기간", "평가 방식", "검증 행", "고장 비율", "Precision", "Recall", "F1", "AP", "탐지 사건", "첫 경보 중앙(분)", "10분 오경보"]);
header(comparison, "A4:K4");
let row = 5;
for (const hours of [12, 24]) {
  const old = replay.results.find(r => r.horizon_hours === hours);
  const current = calendar.results.find(r => r.horizon_hours === hours);
  put(comparison, row++, [`${hours}시간`, "과거: 같은 가동 구간 완전 관측", old.validation_rows,
    old.positive_rows / old.validation_rows, old.precision, old.recall, old.f1,
    old.average_precision, `${old.detected_events}/${old.events}`, old.median_lead_minutes,
    old.false_alert_episodes_10m_cooldown]);
  put(comparison, row++, [`${hours}시간`, "현재: 실제 시계시간", current.validation_rows,
    current.positive_rows / current.validation_rows, current.precision, current.recall,
    current.f1, current.average_precision, `${current.detected_fault_events}/${current.logged_fault_events}`,
    current.first_alert_lead_median_minutes, current.false_alarm_events_10m]);
}
comparison.getRange("C5:C8").setNumberFormat("#,##0");
comparison.getRange("D5:H8").setNumberFormat("0.00%");
comparison.getRange("J5:J8").setNumberFormat("0.0");
comparison.getRange("K5:K8").setNumberFormat("#,##0");
put(comparison, 11, ["해석", "과거 점수가 높아도 현재 시계시간 12/24시간 인수인계 성능으로 해석할 수 없습니다."]);
comparison.getRange("B:B").format.columnWidth = 39;
comparison.freezePanes.freezeRows(4);

put(definition, 2, ["재현 조건과 한계"]);
put(definition, 4, ["항목", "내용"]); header(definition, "A4:B4");
const notes = [
  ["원본", "실제 train 데이터 20개 설비, 원본 4초 센서 82,189,440행 전부를 1분 특징 계산에 사용"],
  ["학습 비율", "1% 실험 제외. 각 기간 XGBoost를 유효한 학습 행 100%로 학습"],
  ["학습/검증", "설비별 시간 순 앞 80%/뒤 20%, 경계 전 최대 예측기간 24시간 purge"],
  ["이전 파이프라인", "삭제된 LS_team6/pipeline.py를 이전 대화의 FileChange 기록 30건에서 복원"],
  ["입력", "현재 센서·공정, 과거 rolling/diff/slope/EWMA. 원본 TTF·고장시각·절대시간은 제외"],
  ["과거 정답", "해당 고장이 같은 연속 가동 구간에서 확인되고 H시간 이내면 위험 1"],
  ["과거 정상", "같은 가동 구간 안에서 H시간 이후까지 관찰할 수 있을 때만 정상 0"],
  ["제외", "위 두 조건을 충족하지 못한 행은 -1로 검열해 학습·평가에서 제외"],
  ["중요한 편향", "가동 구간의 미래 종료 시각을 입력으로 쓰진 않았지만, 이를 기준으로 평가 행을 골랐으므로 실제 운영 모집단과 다름"],
  ["경보 선행", "12/24시간 모델에서 실제 고장 12/24시간 이상 전 경보 사건 0건; 중앙값 약 28분 전"],
  ["범위", "이번 재현은 XGBoost 100% 분류만. HGB는 Windows 실행 정책으로 모듈 로드 불가; TTF 회귀 원본 모델 미복구"],
  ["보고서 상태", "이 파일은 삭제된 원본 엑셀의 바이트 복사가 아니라 원본 코드·실제 데이터로 다시 실행한 새 검증보고서"],
];
notes.forEach((n, i) => put(definition, 5 + i, n));
definition.getRange("B:B").format.columnWidth = 116;

wb.recalculate();
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(target);
console.log(`saved ${target}`);
