const ui = Object.fromEntries(['live-panel','panel-mount','start','end','float','export','ai-options','ai-switch','assess-switch','bleed-switch','session-state','status','interviewer-preview','me-preview','count','log'].map(id => [id, document.getElementById(id)]));
const storeKey = 'atlas-live-session';
let session = { startedAt: null, endedAt: null, entries: [] };
try { session = JSON.parse(sessionStorage.getItem(storeKey)) || session; } catch {}
if (session.startedAt && !session.endedAt) session.endedAt = new Date().toISOString();
let displayStream, microphoneStream, audioContext, audioNodes = [], sockets = {};
let running = false, starting = false, stopping = false, aiSince = 0, aiEpoch = 0, answerEpoch = 0;
let assessSince = 0, assessEpoch = 0, sessionEpoch = 0, assessmentSeq = 0;
let interviewerStartedAt = 0, meStartedAt = 0, interviewerBuffer = [], questionJobs = Promise.resolve(), answerJobs = Promise.resolve(), activeAnswerController;
let assessmentTimer, assessmentJob = Promise.resolve(), pendingMeEntries = [], activeAssessmentController, questionAnalysisControllers = new Set();
let answerJobById = new Map();
let lastPlaybackAt = 0;
let pipWindow = null;

const clock = iso => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const answerVisible = () => ui['ai-switch'].checked || ui['assess-switch'].checked;
function persist() { sessionStorage.setItem(storeKey, JSON.stringify(session)); }
function status(text, error = false) { ui.status.textContent = text; ui.status.classList.toggle('error', error); }
function buttons() {
  ui.start.disabled = running || starting || stopping;
  ui.end.disabled = !running;
  ui.float.disabled = !running && !session.startedAt;
  ui.export.disabled = stopping || !session.entries.length;
  ui['ai-switch'].disabled = !running && (!session.startedAt || starting || stopping);
  ui['assess-switch'].disabled = !running && (!session.startedAt || starting || stopping);
  ui['session-state'].textContent = running ? '录制中' : session.startedAt ? '已结束' : '未开始';
  ui['session-state'].classList.toggle('active', running);
  ui.count.textContent = `${session.entries.length} 条`;
}

function renderEntry(entry) {
  const card = document.createElement('article');
  card.className = `entry ${entry.role}`;
  card.dataset.id = entry.id;
  const head = document.createElement('div'); head.className = 'entry-head';
  const name = document.createElement('b'); name.textContent = entry.role === 'me' ? '我 · 麦克风' : '面试官 · 电脑声音';
  const time = document.createElement('time'); time.textContent = clock(entry.at);
  head.append(name, time);
  const text = document.createElement('div'); text.className = 'entry-text'; text.textContent = entry.text;
  card.append(head, text);
  if (entry.role === 'interviewer' && (entry.aiEligible || entry.assessmentQuestionEligible) && answerVisible() && entry.aiState) {
    const decision = document.createElement('div'); decision.className = 'source-line';
    decision.textContent = ({ ai_off: '仅记录 · AI 已关闭', pre_switch: '仅记录 · 这段话在 AI 开启前开始', judging: '正在判断是否为面试问题…', not_question: '未识别为问题', judge_error: '问题判断失败', answer_error: '回答失败', queued: '已识别为问题 · 等待回答', answering: '正在生成回答…', answered: '已回答', stopped: 'AI 回答已停止' })[entry.aiState] || '';
    card.append(decision);
    if (!entry.question && answerVisible() && ['not_question','judge_error'].includes(entry.aiState)) {
      const manual = document.createElement('button'); manual.type = 'button'; manual.className = 'manual-answer';
      manual.textContent = '按这段话回答'; manual.onclick = () => answerManually(entry);
      card.append(manual);
    }
  }
  if (answerVisible() && entry.answer !== undefined) {
    const box = document.createElement('div'); box.className = 'answer-box';
    const title = document.createElement('b'); title.textContent = entry.question ? `AI 回答 · ${entry.question}` : 'AI 回答';
    const body = document.createElement('div'); body.textContent = entry.answer || '正在生成…';
    box.append(title);
    if (entry.sourceMode === 'model_fallback') {
      const tag = document.createElement('span'); tag.className = 'answer-tag';
      tag.textContent = entry.answerTag || '模型经验回答 · 未经资料核实';
      box.append(tag);
    }
    box.append(body);
    if (entry.sources?.length) {
      const sources = document.createElement('div'); sources.className = 'source-line';
      sources.textContent = `来源：${entry.sources.map(s => s.title || s.source).join('；')}`;
      box.append(sources);
    }
    card.append(box);
  }
  if (ui['assess-switch'].checked && entry.questionAnalysis) {
    const analysis = entry.questionAnalysis;
    const box = assessmentBox('面试官问题分析', analysis.verification_tag);
    appendLine(box, '考点', analysis.focus);
    appendPoints(box, '应答要点', analysis.expected_points);
    appendLine(box, '', analysis.error || (analysis.pending ? '正在分析问题…' : ''));
    card.append(box);
  }
  if (ui['assess-switch'].checked && entry.assessment) {
    const assessment = entry.assessment;
    const verdicts = { correct: '正确', partial: '部分正确', incorrect: '需要纠正', uncertain: '暂无法判断' };
    const box = assessmentBox(`我的回答分析${assessment.verdict ? ` · ${verdicts[assessment.verdict] || assessment.verdict}` : ''}`, assessment.verification_tag);
    appendLine(box, '正确性', assessment.correctness);
    appendLine(box, '完整度', assessment.completeness);
    appendLine(box, '说明', assessment.explanation);
    appendPoints(box, '遗漏要点', assessment.missing_points);
    appendLine(box, '补充回答', assessment.supplement);
    appendLine(box, '', assessment.error || (assessment.pending ? '正在分析回答…' : ''));
    card.append(box);
  }
  return card;
}
function assessmentBox(title, verificationTag) {
  const box = document.createElement('div'); box.className = 'assessment-box';
  const heading = document.createElement('b'); heading.textContent = title; box.append(heading);
  if (verificationTag) {
    const tag = document.createElement('span');
    tag.className = `answer-tag${verificationTag.includes('未经') || verificationTag.includes('Unverified') ? ' unverified' : ''}`;
    tag.textContent = verificationTag; box.append(tag);
  }
  return box;
}
function appendLine(box, label, value) {
  if (!value) return;
  const line = document.createElement('p'); line.textContent = label ? `${label}：${value}` : value; box.append(line);
}
function appendPoints(box, label, points) {
  if (!points?.length) return;
  const title = document.createElement('p'); title.textContent = `${label}：`; box.append(title);
  const list = document.createElement('ul');
  for (const point of points) { const item = document.createElement('li'); item.textContent = point; list.append(item); }
  box.append(list);
}
function renderLog() {
  ui.log.replaceChildren();
  if (!session.entries.length) { const empty = document.createElement('p'); empty.className = 'empty'; empty.textContent = '会话记录会显示在这里。'; ui.log.append(empty); }
  else session.entries.forEach(entry => ui.log.append(renderEntry(entry)));
  buttons();
}
function addTranscript(role, text) {
  if (!text.trim()) return null;
  const entry = { id: crypto.randomUUID(), role, at: new Date().toISOString(), text: text.trim() };
  session.entries.push(entry); persist(); renderLog();
  ui.log.scrollTop = ui.log.scrollHeight;
  return entry;
}

async function classifyAndAnswer(entry, generation) {
  if (generation !== sessionEpoch) return;
  interviewerBuffer.push(entry.text);
  interviewerBuffer = interviewerBuffer.slice(-3);
  let response;
  try {
    response = await fetch('/api/live/question', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: interviewerBuffer.join(' ').slice(-1200) }) });
    if (!response.ok) throw Error('问题判断失败');
    response = await response.json();
  } catch (error) { if (generation !== sessionEpoch) return; entry.aiState = 'judge_error'; if (entry.assessmentQuestionEligible) entry.questionAnalysis = { error: '问题判断失败' }; persist(); renderLog(); status(error.message, true); return; }
  if (generation !== sessionEpoch) return;
  if (!response.question) { entry.aiState = 'not_question'; delete entry.questionAnalysis; persist(); renderLog(); return; }
  interviewerBuffer = [];
  entry.question = response.question;
  if ((entry.aiEligible && ui['ai-switch'].checked && entry.aiEpoch === aiEpoch) ||
      (entry.assessmentQuestionEligible && ui['assess-switch'].checked && entry.assessEpoch === assessEpoch)) {
    entry.answer = ''; entry.aiState = 'queued';
    answerJobs = answerJobs.then(() => generateAnswer(entry, answerEpoch));
    answerJobById.set(entry.id, answerJobs);
  }
  if (entry.assessmentQuestionEligible && ui['assess-switch'].checked && entry.assessEpoch === assessEpoch) {
    entry.questionAnalysis = { pending: true };
    analyzeQuestion(entry, generation, entry.assessEpoch, answerJobs);
  }
  persist(); renderLog();
}

async function analyzeQuestion(entry, generation, epoch, referenceJob) {
  const controller = new AbortController(); questionAnalysisControllers.add(controller);
  const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    await referenceJob;
    if (generation !== sessionEpoch || epoch !== assessEpoch || controller.signal.aborted) return;
    if (entry.aiState !== 'answered' || !entry.answer) {
      entry.questionAnalysis = { error: '问题回答未完成，暂无法分析考点' };
      persist(); renderLog(); return;
    }
    const reference = { reference_answer: entry.answer.slice(0, 8000), reference_mode: entry.sourceMode };
    const response = await fetch('/api/live/analyze-question', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: entry.question, ...reference }), signal: controller.signal
    });
    if (!response.ok) throw Error('问题分析请求失败');
    const result = await response.json();
    if (generation !== sessionEpoch || epoch !== assessEpoch) return;
    entry.questionAnalysis = result; persist(); renderLog();
  } catch (error) {
    if (generation !== sessionEpoch || epoch !== assessEpoch) return;
    entry.questionAnalysis = { error: error.name === 'AbortError' ? '问题分析超时' : error.message };
    persist(); renderLog();
  } finally { clearTimeout(timeout); questionAnalysisControllers.delete(controller); }
}

function answerManually(entry) {
  if (!answerVisible() || entry.question) return;
  entry.question = entry.text;
  entry.answer = '';
  entry.aiState = 'queued';
  interviewerBuffer = [];
  persist(); renderLog();
  const epoch = answerEpoch;
  answerJobs = answerJobs.then(() => generateAnswer(entry, epoch));
  answerJobById.set(entry.id, answerJobs);
  if (ui['assess-switch'].checked && entry.assessmentQuestionEligible && entry.assessEpoch === assessEpoch) {
    entry.questionAnalysis = { pending: true };
    analyzeQuestion(entry, sessionEpoch, assessEpoch, answerJobs);
  }
}

async function generateAnswer(entry, epoch) {
  if (!answerVisible() || epoch !== answerEpoch) { entry.aiState = 'stopped'; persist(); renderLog(); return; }
  entry.aiState = 'answering'; persist(); renderLog();
  activeAnswerController = new AbortController();
  const controller = activeAnswerController;
  const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch('/api/ask/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: entry.question, force_knowledge_base: true }), signal: controller.signal });
    if (!response.ok || !response.body) throw Error('回答请求失败');
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 1);
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === 'delta') { entry.answer += event.text; renderLog(); }
        if (event.type === 'error') throw Error(event.message);
        if (event.type === 'done') { entry.answer = event.result.answer; entry.sources = event.result.sources || []; entry.sourceMode = event.result.source_mode; entry.answerTag = event.result.answer_tag; entry.aiState = 'answered'; persist(); renderLog(); return; }
      }
    }
    throw Error('回答连接中断');
  } catch (error) {
    entry.answer = error.name === 'AbortError' ? 'AI 回答已停止或超时。' : `回答失败：${error.message}`;
    entry.aiState = error.name === 'AbortError' ? 'stopped' : 'answer_error';
    persist(); renderLog();
  } finally { clearTimeout(timeout); if (activeAnswerController === controller) activeAnswerController = null; }
}

function onAsrEvent(role, event) {
  if (event.type === 'partial') {
    ui[`${role}-preview`].textContent = event.text || '识别中…';
    if (role === 'interviewer' && !interviewerStartedAt) interviewerStartedAt = Date.now();
    if (role === 'me' && !meStartedAt) meStartedAt = Date.now();
    return;
  }
  if (event.type === 'speech_started') {
    if (role === 'interviewer' && !interviewerStartedAt) interviewerStartedAt = Date.now();
    if (role === 'me' && !meStartedAt) meStartedAt = Date.now();
    return;
  }
  if (event.type === 'error') { status(`${role === 'me' ? '麦克风' : '面试官'}识别失败：${event.message}`, true); if (running) endSession(); return; }
  if (event.type !== 'transcript') return;
  ui[`${role}-preview`].textContent = event.text || '等待语音…';
  const entry = addTranscript(role, event.text || '');
  if (role === 'interviewer' && entry) {
    const beganAt = interviewerStartedAt || Date.now();
    entry.aiEligible = ui['ai-switch'].checked && beganAt >= aiSince;
    entry.assessmentQuestionEligible = ui['assess-switch'].checked && beganAt >= assessSince;
    entry.aiEpoch = aiEpoch; entry.assessEpoch = assessEpoch;
    entry.aiState = !entry.aiEligible && !entry.assessmentQuestionEligible ? 'pre_switch' : 'judging';
    persist(); renderLog();
    if (entry.aiEligible || entry.assessmentQuestionEligible) {
      const generation = sessionEpoch;
      questionJobs = questionJobs.then(() => classifyAndAnswer(entry, generation)).catch(error => status(`问题处理失败：${error.message}`, true));
    }
  }
  if (role === 'me' && entry) {
    const beganAt = meStartedAt || Date.now();
    entry.assessmentEligible = ui['assess-switch'].checked && beganAt >= assessSince;
    entry.assessEpoch = assessEpoch;
    if (entry.assessmentEligible) { pendingMeEntries.push(entry); scheduleAssessment(); }
  }
  if (role === 'interviewer') interviewerStartedAt = 0;
  if (role === 'me') meStartedAt = 0;
}

function scheduleAssessment(delay = 2500) {
  clearTimeout(assessmentTimer);
  assessmentTimer = setTimeout(() => { assessmentJob = assessPendingAnswers().catch(error => status(`回答分析失败：${error.message}`, true)); }, delay);
}

async function assessPendingAnswers() {
  if (!pendingMeEntries.length || !ui['assess-switch'].checked) return;
  const entries = pendingMeEntries.splice(0);
  const epoch = assessEpoch, generation = sessionEpoch;
  await questionJobs;
  if (epoch !== assessEpoch || generation !== sessionEpoch || !ui['assess-switch'].checked) return;
  const firstIndex = session.entries.indexOf(entries[0]);
  const questionEntry = session.entries.slice(0, firstIndex).reverse().find(item => item.role === 'interviewer' && item.question && item.assessmentQuestionEligible && item.assessEpoch === epoch);
  if (!questionEntry) return;
  await (answerJobById.get(questionEntry.id) || Promise.resolve());
  if (epoch !== assessEpoch || generation !== sessionEpoch || !ui['assess-switch'].checked) return;
  const questionIndex = session.entries.indexOf(questionEntry);
  const target = entries.at(-1);
  const answer = session.entries.slice(questionIndex + 1, session.entries.indexOf(target) + 1)
    .filter(item => item.role === 'me' && item.assessmentEligible && item.assessEpoch === epoch)
    .map(item => item.text).join(' ').slice(-4000);
  if (!answer) return;
  for (const item of session.entries.slice(questionIndex + 1, session.entries.indexOf(target))) {
    if (item.role === 'me') delete item.assessment;
  }
  if (activeAssessmentController) activeAssessmentController.abort();
  const sequence = ++assessmentSeq;
  const controller = new AbortController(); activeAssessmentController = controller;
  target.assessment = { pending: true }; persist(); renderLog();
  const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const reference = questionEntry.questionAnalysis?.reference_answer
      ? { reference_answer: questionEntry.questionAnalysis.reference_answer.slice(0, 8000), reference_mode: questionEntry.questionAnalysis.reference_mode }
      : questionEntry.aiState === 'answered' && questionEntry.answer
        ? { reference_answer: questionEntry.answer.slice(0, 8000), reference_mode: questionEntry.sourceMode }
        : {};
    const response = await fetch('/api/live/assess', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: questionEntry.question, answer, ...reference }), signal: controller.signal
    });
    if (!response.ok) throw Error('回答分析请求失败');
    const result = await response.json();
    if (epoch !== assessEpoch || generation !== sessionEpoch || sequence !== assessmentSeq) return;
    target.assessment = result; persist(); renderLog();
  } catch (error) {
    if (epoch !== assessEpoch || generation !== sessionEpoch || sequence !== assessmentSeq) return;
    target.assessment = { error: error.name === 'AbortError' ? '回答分析超时' : error.message };
    persist(); renderLog();
  } finally { clearTimeout(timeout); if (activeAssessmentController === controller) activeAssessmentController = null; }
}

function openAsr(role) {
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/api/live/asr/${role}`);
    sockets[role] = socket;
    let ready = false;
    const timeout = setTimeout(() => { if (!ready) { socket.close(); reject(Error('实时识别连接超时')); } }, 15000);
    socket.onmessage = message => {
      if (sockets[role] !== socket) return;
      let event; try { event = JSON.parse(message.data); } catch { return; }
      if (event.type === 'ready') { ready = true; clearTimeout(timeout); resolve(socket); }
      else if (event.type === 'error' && !ready) { clearTimeout(timeout); reject(Error(event.message)); }
      else onAsrEvent(role, event);
    };
    socket.onerror = () => { if (!ready) { clearTimeout(timeout); reject(Error('实时识别连接失败')); } };
    socket.onclose = () => {
      clearTimeout(timeout);
      if (!ready) reject(Error('实时识别连接已关闭'));
      else if (running && !stopping && sockets[role] === socket) { status(`${role === 'me' ? '麦克风' : '面试官'}识别连接中断，请结束后重试。`, true); endSession(); }
    };
  });
}

const processorCode = `class PcmProcessor extends AudioWorkletProcessor {
  constructor(){super();this.input=[];this.pending=[];}
  process(inputs){
    const channels=inputs[0]; if(!channels||!channels[0])return true;
    for(let i=0;i<channels[0].length;i++)this.input.push(channels.length>1?(channels[0][i]+channels[1][i])/2:channels[0][i]);
    const ratio=sampleRate/16000,count=Math.floor((this.input.length-1)/ratio);
    for(let i=0;i<count;i++){
      const at=i*ratio,left=Math.floor(at),mix=at-left;
      const value=Math.max(-1,Math.min(1,this.input[left]*(1-mix)+this.input[left+1]*mix));
      this.pending.push(value<0?value*32768:value*32767);
    }
    this.input.splice(0,Math.floor(count*ratio));
    while(this.pending.length>=3200){
      const pcm=new Int16Array(this.pending.splice(0,3200));
      let power=0;for(const sample of pcm)power+=sample*sample;
      this.port.postMessage({buffer:pcm.buffer,rms:Math.sqrt(power/pcm.length)/32768},[pcm.buffer]);
    }
    return true;
  }
} registerProcessor('pcm-processor',PcmProcessor);`;

async function connectAudio(role, stream) {
  const source = audioContext.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
  const processor = new AudioWorkletNode(audioContext, 'pcm-processor');
  const silent = audioContext.createGain(); silent.gain.value = 0;
  source.connect(processor).connect(silent).connect(audioContext.destination);
  processor.port.onmessage = message => {
    const socket = sockets[role];
    if (role === 'interviewer' && message.data.rms > 0.006) lastPlaybackAt = Date.now();
    if (socket?.readyState !== WebSocket.OPEN || socket.bufferedAmount >= 256000) return;
    const crosstalk = role === 'me' && ui['bleed-switch'].checked && Date.now() - lastPlaybackAt < 250;
    socket.send(crosstalk ? new Int16Array(message.data.buffer.byteLength / 2).buffer : message.data.buffer);
  };
  audioNodes.push(source, processor, silent);
}

async function releaseAudio() {
  for (const node of audioNodes) { try { node.disconnect(); } catch {} }
  audioNodes = [];
  if (audioContext) { await audioContext.close(); audioContext = null; }
  for (const stream of [displayStream, microphoneStream]) stream?.getTracks().forEach(track => track.stop());
  displayStream = microphoneStream = null;
}

async function startSession() {
  if (running || starting) return;
  starting = true; stopping = false; buttons();
  status('请选择会议窗口或整个屏幕，并勾选共享音频…');
  try {
    if (!navigator.mediaDevices?.getDisplayMedia || !window.AudioWorkletNode) throw Error('请使用支持共享音频的桌面版 Chrome 或 Edge。');
    for (const socket of Object.values(sockets)) socket.close();
    sockets = {};
    displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true, systemAudio: 'include', windowAudio: 'window', selfBrowserSurface: 'exclude' });
    if (!displayStream.getAudioTracks().length) throw Error('未取得会议播放音轨。请重新选择会议窗口或屏幕，并勾选共享音频。');
    status('请允许访问麦克风…');
    microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    const micTrack = microphoneStream.getAudioTracks()[0];
    const cancellation = micTrack.getCapabilities?.().echoCancellation;
    if (Array.isArray(cancellation) && cancellation.includes('all')) {
      try { await micTrack.applyConstraints({ echoCancellation: 'all', noiseSuppression: true }); } catch {}
    }
    session = { startedAt: new Date().toISOString(), endedAt: null, entries: [] };
    sessionEpoch++; aiEpoch++; assessEpoch++; answerEpoch++; assessmentSeq++;
    clearTimeout(assessmentTimer); pendingMeEntries = [];
    activeAnswerController?.abort(); activeAssessmentController?.abort();
    for (const controller of questionAnalysisControllers) controller.abort();
    questionAnalysisControllers.clear();
    persist(); renderLog(); interviewerBuffer = []; aiSince = 0; assessSince = 0;
    questionJobs = Promise.resolve(); answerJobs = Promise.resolve();
    assessmentJob = Promise.resolve();
    answerJobById = new Map();
    ui['ai-switch'].checked = false; ui['assess-switch'].checked = false;
    ui['ai-options'].open = false;
    interviewerStartedAt = 0; meStartedAt = 0; lastPlaybackAt = 0;
    status('正在连接两路实时语音识别…');
    await Promise.all([openAsr('interviewer'), openAsr('me')]);
    audioContext = new AudioContext();
    const url = URL.createObjectURL(new Blob([processorCode], { type: 'text/javascript' }));
    try { await audioContext.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
    await connectAudio('interviewer', displayStream);
    await connectAudio('me', microphoneStream);
    await audioContext.resume();
    displayStream.getVideoTracks().forEach(track => { track.onended = () => { if (running) endSession(); }; });
    running = true; starting = false; buttons();
    status(`正在记录双方语音。麦克风：${micTrack.label || '默认设备'}。`);
  } catch (error) {
    await releaseAudio();
    for (const socket of Object.values(sockets)) socket.close(); sockets = {};
    if (session.startedAt && !session.endedAt) { session.endedAt = new Date().toISOString(); persist(); }
    starting = false; buttons(); status(error.message, true);
  }
}

async function endSession() {
  if ((!running && !starting) || stopping) return;
  stopping = true; running = false; starting = false;
  await releaseAudio();
  buttons(); status('正在整理最后的语音…');
  await Promise.all(Object.values(sockets).map(socket => new Promise(resolve => {
    if (socket.readyState === WebSocket.CLOSED) { resolve(); return; }
    const timer = setTimeout(() => { socket.close(); resolve(); }, 5500);
    socket.addEventListener('close', () => { clearTimeout(timer); resolve(); }, { once: true });
    if (socket.readyState === WebSocket.OPEN) socket.send('stop');
  })));
  clearTimeout(assessmentTimer);
  if (pendingMeEntries.length) assessmentJob = assessPendingAnswers();
  await assessmentJob;
  session.endedAt = new Date().toISOString(); persist(); stopping = false; buttons();
  status('会话已结束。记录可导出；开始新会话会覆盖当前记录。');
}

function exportSession() {
  if (!session.entries.length) return;
  const lines = ['# Atlas 通话记录', '', `开始：${session.startedAt}`, `结束：${session.endedAt || '进行中'}`, ''];
  for (const entry of session.entries) {
    lines.push(`## ${clock(entry.at)} ${entry.role === 'me' ? '我' : '面试官'}`, '', entry.text, '');
    if (entry.question) lines.push(`**识别问题：** ${entry.question}`, '');
    if (entry.answer !== undefined) lines.push(`**AI 回答：** ${entry.answer || '未完成'}`, '');
    if (entry.questionAnalysis && !entry.questionAnalysis.pending) {
      const analysis = entry.questionAnalysis;
      if (analysis.focus) lines.push(`**面试官问题考点：** ${analysis.focus}`, '');
      if (analysis.expected_points?.length) lines.push('**应答要点：**', ...analysis.expected_points.map(point => `- ${point}`), '');
      if (analysis.verification_tag) lines.push(`**分析标识：** ${analysis.verification_tag}`, '');
    }
    if (entry.assessment && !entry.assessment.pending) {
      const review = entry.assessment;
      if (review.verdict) lines.push(`**我的回答判断：** ${review.verdict}`, '');
      if (review.correctness) lines.push(`**正确性：** ${review.correctness}`, '');
      if (review.completeness) lines.push(`**完整度：** ${review.completeness}`, '');
      if (review.explanation) lines.push(`**说明：** ${review.explanation}`, '');
      if (review.missing_points?.length) lines.push('**遗漏要点：**', ...review.missing_points.map(point => `- ${point}`), '');
      if (review.supplement) lines.push(`**补充回答：** ${review.supplement}`, '');
      if (review.verification_tag) lines.push(`**分析标识：** ${review.verification_tag}`, '');
    }
    if (entry.sourceMode === 'model_fallback') lines.push(`**回答标识：** ${entry.answerTag || '模型经验回答 · 未经资料核实'}`, '');
    for (const source of entry.sources || []) lines.push(`- 来源：${source.title || ''} ${source.source || ''}`);
    if (entry.sources?.length) lines.push('');
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = `atlas-interview-${new Date(session.startedAt).toISOString().slice(0, 19).replaceAll(':', '-')}.md`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function toggleFloat() {
  if (pipWindow) { pipWindow.close(); return; }
  if (!('documentPictureInPicture' in window)) { status('当前浏览器不支持置顶悬浮窗，请使用新版 Chrome 或 Edge。', true); return; }
  try {
    pipWindow = await documentPictureInPicture.requestWindow({ width: 430, height: 620 });
    const link = pipWindow.document.createElement('link'); link.rel = 'stylesheet'; link.href = '/static/live.css';
    pipWindow.document.head.append(link);
    pipWindow.document.body.className = 'pip-body';
    pipWindow.document.body.append(ui['live-panel']);
    ui.float.textContent = '返回页面';
    pipWindow.addEventListener('pagehide', () => { ui['panel-mount'].append(ui['live-panel']); pipWindow = null; ui.float.textContent = '悬浮小窗'; });
  } catch (error) { status(`无法打开悬浮窗：${error.message}`, true); }
}

ui.start.onclick = startSession;
ui.end.onclick = endSession;
ui.float.onclick = toggleFloat;
ui.export.onclick = exportSession;
ui['ai-switch'].onchange = () => {
  aiEpoch++; interviewerBuffer = [];
  if (!answerVisible()) { answerEpoch++; activeAnswerController?.abort(); }
  aiSince = running && ui['ai-switch'].checked ? Date.now() : 0;
  renderLog();
  status(running
    ? (ui['ai-switch'].checked ? 'AI 回答已开启：回答新听到的面试官问题。' : ui['assess-switch'].checked ? 'AI 回答单独开关已关闭；双向分析仍会回答面试官问题。' : 'AI 回答已关闭：答案已隐藏，继续记录双方语音。')
    : (answerVisible() ? '已显示本次会话的 AI 答案。' : '已隐藏本次会话的 AI 答案。'));
};
ui['assess-switch'].onchange = () => {
  assessEpoch++; assessmentSeq++; interviewerBuffer = [];
  clearTimeout(assessmentTimer); pendingMeEntries = [];
  activeAssessmentController?.abort();
  if (!answerVisible()) { answerEpoch++; activeAnswerController?.abort(); }
  for (const controller of questionAnalysisControllers) controller.abort();
  questionAnalysisControllers.clear();
  if (!ui['assess-switch'].checked) {
    for (const entry of session.entries) {
      if (entry.questionAnalysis?.pending) delete entry.questionAnalysis;
      if (entry.assessment?.pending) delete entry.assessment;
    }
    persist();
  }
  assessSince = running && ui['assess-switch'].checked ? Date.now() : 0;
  renderLog();
  status(running
    ? (ui['assess-switch'].checked ? '双向分析已开启：回答新听到的面试官问题，并评价、补充你随后说出的回答。' : '双向分析已关闭：分析内容已隐藏，继续记录双方语音。')
    : (ui['assess-switch'].checked ? '已显示本次会话的双向分析。' : '已隐藏本次会话的双向分析。'));
};
persist(); renderLog();
