import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const base = path.resolve('output/experiments/model_validation_decision_12h_24h');
const experiments = path.resolve('output/experiments');
const readCsv = async file => {
  const text = (await fs.readFile(file, 'utf8')).replace(/^\uFEFF/, '').trim();
  const [head, ...lines] = text.split(/\r?\n/).map(x => x.split(','));
  return lines.map(values => Object.fromEntries(head.map((h, i) => [h, values[i] ?? ''])));
};
const num = v => v === '' || v == null ? null : Number(v);
const classification = await readCsv(path.join(base, 'classification.csv'));
const regression = await readCsv(path.join(base, 'regression.csv'));
const policy = await readCsv(path.join(base, 'alarm_policy_comparison.csv'));
const grid12 = await readCsv(path.join(experiments, 'threshold_persistence_grid_12h/grid.csv'));
const byTool = await readCsv(path.join(experiments, 'threshold_persistence_grid_24h/false_alarm_by_tool.csv'));
const doc = JSON.parse(await fs.readFile(path.join(base, 'decision.json'), 'utf8'));

const wb = Workbook.create();
const navy = '#17365D', pale = '#EAF1F8', red = '#FCE8E6', amber = '#FFF2D5';
const make = name => {
  const s = wb.worksheets.add(name);
  s.showGridLines = false;
  s.getRange('A1:Q40').format.font = {name:'Arial', size:10, color:'#202C3A'};
  s.getRange('A:Q').format.columnWidth = 15;
  s.getRange('A:A').format.columnWidth = 20;
  return s;
};
const title = (s, text) => {
  s.getRange('A2').values = [[text]];
  s.getRange('A2').format.font = {name:'Arial', size:15, bold:true, color:navy};
  s.getRange('A3:J3').format.borders = {preset:'doubleBottom', color:'#B8C8D9'};
};
const header = (s, range) => {
  s.getRange(range).format = {fill:navy, font:{name:'Arial', size:10, bold:true, color:'#FFFFFF'}, rowHeight:27,
    horizontalAlignment:'center', verticalAlignment:'center'};
};
const write = (s, address, matrix) => s.getRange(address).write(matrix);

// Executive view: fixed historical validation, with the deployment gate explicit.
{
  const s = make('결론'); title(s, '12h·24h 모델 검증 결과');
  s.getRange('A:A').format.columnWidth = 24;
  s.getRange('B:B').format.columnWidth = 44;
  s.getRange('C:C').format.columnWidth = 21;
  s.getRange('D:D').format.columnWidth = 22;
  write(s, 'A5', [
    ['선정한 분류 모델', 'XGBoost 12h / 24h'],
    ['TTF 회귀 모델', '선정 보류 — HGB-log·LSTM 모두 R² 음수'],
    ['24h 경보 후보', '점수 ≥ 0.36, 최근 5분 중 2회'],
    ['대시보드 연결', '아직 진행하지 않음'],
  ]);
  s.getRange('A5:A8').format.font = {name:'Arial', size:10, bold:true, color:navy};
  s.getRange('B6').format.fill = amber;
  s.getRange('B8').format.fill = pale;
  write(s, 'A11', [['24h 후보 지표', '값', '판단 기준', '해석']]); header(s, 'A11:D11');
  write(s, 'A12', [
    ['고장 사건 탐지', 138, '전체 181건', '기존 0.5 단독 112건'],
    ['10분 묶음 오경보', 6066, '기존 3,414건', '2,652건 증가'],
    ['설비당 12h 평균 오경보', 3.824073295788855, '허용 약 4건', '관측분 환산 평균'],
    ['허용치 초과 설비', 9, '20대 중', '운영 조건 미충족'],
    ['최다 설비 12h 오경보', 13.641129403108618, '허용 약 4건', '운영 조건 미충족'],
  ]);
  s.getRange('B14:B16').setNumberFormat('0.00');
  s.getRange('A15:D16').format.fill = red;
  s.getRange('A19').values = [['검증 한계']];
  s.getRange('A19').format.font = {name:'Arial', size:11, bold:true, color:navy};
  write(s, 'A20', [
    ['임계값과 2/5회 조건을 선택한 검증 구간을 재사용했습니다. 이후 시간 구간의 독립 검증이 필요합니다.'],
    ['12h 오경보율은 관측된 1분 행을 720분으로 나눈 환산치이며 실제 근무조별 횟수와 다릅니다.'],
    ['고장 유형·TTF는 현재 모델 성능으로 표시하거나 운영 판단에 쓰지 않습니다.'],
  ]);
  s.getRange('A20:A22').format.columnWidth = 105;
}

// Same validation population and 0.5 threshold for an apples-to-apples model comparison.
{
  const s = make('분류 비교'); title(s, 'HGB와 XGBoost · 동일 검증 구간');
  s.getRange('A5:N5').values = [['모델','예측시간(h)','AP','Precision','Recall','F1','Balanced Acc.','탐지/전체','평가가능 사건','오경보(10분)','TN','FP','FN','TP']];
  header(s, 'A5:N5');
  const rows = classification.map(r => [r.model,num(r.horizon_hours),num(r.average_precision),num(r.precision),num(r.recall),num(r.f1),num(r.balanced_accuracy),
    `${r.detected_fault_events}/${r.logged_fault_events}`,num(r.evaluable_fault_events),num(r.false_alarm_events_10m),num(r.tn),num(r.fp),num(r.fn),num(r.tp)]);
  write(s, 'A6', rows);
  s.getRange('C6:G9').setNumberFormat('0.00%');
  s.getRange('I6:N9').setNumberFormat('#,##0');
  s.getRange('H:H').format.columnWidth = 18;
  s.getRange('J:N').format.columnWidth = 17;
  s.getRange('A12').values = [['비교 조건']];
  write(s, 'A13', [['20대 설비, 85개 입력변수, 시간순 80/20 분할, 24h purge, 동일 1,142,112개 검증 행.']]);
  write(s, 'A14', [['양성 비율이 달라 12h와 24h의 AP 수치 자체를 직접 우열 비교하지 않습니다.']]);
  s.getRange('A13:A14').format.columnWidth = 97;
}

{
  const s = make('경보 조건'); title(s, '최근 5분 중 2회 경보 비교');
  s.getRange('A5:K5').values = [['모델','예측시간(h)','Threshold','연속 조건','탐지 건','전체 181건 Recall','평가가능 Recall','오경보(10분)','설비당 12h 평균','4건 초과 설비','설비당 최고']];
  header(s, 'A5:K5');
  const x = policy.find(r => r.model==='XGBoost');
  const h = policy.find(r => r.model==='HGB');
  const g12 = grid12.find(r => num(r.threshold)===.36 && r.condition==='2_of_5m');
  const base24 = classification.find(r => r.model==='XGBoost' && num(r.horizon_hours)===24);
  const rows = [
    ['XGBoost',24,.5,'단독',num(base24.detected_fault_events),112/181,112/163,num(base24.false_alarm_events_10m),3414/(1142112/720),5,8.249317179074165],
    [h.model,24,num(h.threshold),'2/5회',num(h.detected_fault_events),num(h.all_fault_event_recall),num(h.evaluable_fault_event_recall),num(h.false_alarm_events_10m),num(h.false_alarm_events_per_observed_tool_12h),num(h.tools_over_4_false_alarms_per_observed_12h),num(h.worst_tool_false_alarms_per_observed_12h)],
    [x.model,24,num(x.threshold),'2/5회',num(x.detected_fault_events),num(x.all_fault_event_recall),num(x.evaluable_fault_event_recall),num(x.false_alarm_events_10m),num(x.false_alarm_events_per_observed_tool_12h),num(x.tools_over_4_false_alarms_per_observed_12h),num(x.worst_tool_false_alarms_per_observed_12h)],
    ['XGBoost',12,.36,'2/5회',num(g12.detected_fault_events),num(g12.all_fault_event_recall),num(g12.evaluable_fault_event_recall),num(g12.false_alarm_events_10m),num(g12.false_alarm_events_per_observed_tool_day)/2,null,null],
  ];
  write(s,'A6',rows);
  s.getRange('C6:C9').setNumberFormat('0.00');
  s.getRange('F6:G9').setNumberFormat('0.0%');
  s.getRange('I6:I9').setNumberFormat('0.00');
  s.getRange('K6:K9').setNumberFormat('0.00');
  s.getRange('A8:K8').format.fill = pale;
  s.getRange('A12').values = [['2/5회: 현재 포함 최근 5개 시계분에서 threshold 이상 관측이 2회 이상. 2분 초과 정지 후 이력을 초기화합니다.']];
  s.getRange('A:A').format.columnWidth = 19;
  s.getRange('D:D').format.columnWidth = 18;
  s.getRange('F:K').format.columnWidth = 19;
}

{
  const s = make('설비별 오경보'); title(s, '24h XGBoost · 설비별 오경보');
  s.getRange('A5:G5').values = [['설비','0.5 단독 사건','0.5 단독 /12h','0.36·2/5 사건','0.36·2/5 /12h','허용치(건)','판정']];
  header(s,'A5:G5');
  const baseline = new Map(byTool.filter(r => num(r.grid_threshold)===.5 && r.condition==='1_of_1m').map(r => [r.tool_id,r]));
  const selected = byTool.filter(r => num(r.grid_threshold)===.36 && r.condition==='2_of_5m').sort((a,b)=>a.tool_id.localeCompare(b.tool_id));
  const rows = selected.map(r => {
    const b = baseline.get(r.tool_id);
    const rate = num(r.false_alarm_events_per_observed_day)/2;
    return [r.tool_id,num(b.false_alarm_events_10m),num(b.false_alarm_events_per_observed_day)/2,
      num(r.false_alarm_events_10m),rate,4,rate>4?'초과':'충족'];
  });
  write(s,'A6',rows);
  s.getRange('C6:C25').setNumberFormat('0.00');
  s.getRange('E6:F25').setNumberFormat('0.00');
  s.getRange('B6:B25').setNumberFormat('#,##0');
  s.getRange('D6:D25').setNumberFormat('#,##0');
  s.getRange('A:A').format.columnWidth = 19;
  s.getRange('B:G').format.columnWidth = 20;
  rows.forEach((r,i)=>{if(r[6]==='초과') s.getRange(`E${i+6}:G${i+6}`).format.fill = red;});
  s.freezePanes.freezeRows(5);
  s.getRange('A28').values = [['/12h는 관측된 1분 행 기준 환산치입니다. 실제 12시간 근무조에서 발생한 횟수는 아닙니다.']];
}

{
  const s = make('TTF 회귀'); title(s, '유형별 잔여시간 예측 · HGB-log와 LSTM');
  s.getRange('A5:F5').values = [['모델','고장 유형','학습 행','검증 행','MAE (시간)','R²']]; header(s,'A5:F5');
  const rows = regression.map(r => [r.model,r.fault_type,num(r.train_rows),num(r.validation_rows),num(r.mae_hours),num(r.r2)]);
  write(s,'A6',rows);
  s.getRange('C6:D11').setNumberFormat('#,##0');
  s.getRange('E6:E11').setNumberFormat('#,##0.0');
  s.getRange('F6:F11').setNumberFormat('0.000');
  s.getRange('F6:F11').format.fill = red;
  s.getRange('A:A').format.columnWidth = 18;
  s.getRange('B:B').format.columnWidth = 22;
  s.getRange('C:F').format.columnWidth = 19;
  s.getRange('A14').values = [['세 유형 모두 R²가 음수입니다. MAE 감소만으로 TTF 모델을 운영용으로 선정하지 않습니다.']];
}

wb.recalculate();
const checks = [
  ['결론','A11:D16'],['분류 비교','A5:N9'],['경보 조건','A5:K9'],
  ['설비별 오경보','A5:G25'],['TTF 회귀','A5:F11'],
];
for (const [sheetName, range] of checks) {
  const check = await wb.inspect({kind:'table', range:`'${sheetName}'!${range}`, include:'values', tableMaxRows:5, tableMaxCols:12, maxChars:1600});
  console.log(sheetName, check.ndjson.slice(0,800));
}
const errors = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:20},maxChars:1000});
console.log('error scan',errors.ndjson);
const file = await SpreadsheetFile.exportXlsx(wb);
const target = path.join(base,'모델검증_12h_24h_HGB_XGBoost_TTF_경보비교.xlsx');
await file.save(target);
console.log('saved',target,'regression rows',regression.length);
