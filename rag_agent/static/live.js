const ui = Object.fromEntries(['live-panel','panel-mount','start','end','float','export','ai-switch','bleed-switch','session-state','status','interviewer-preview','me-preview','count','log'].map(id => [id, document.getElementById(id)]));
const storeKey = 'atlas-live-session';
let session = { startedAt: null, endedAt: null, entries: [] };
try { session = JSON.parse(sessionStorage.getItem(storeKey)) || session; } catch {}
if (session.startedAt && !session.endedAt) session.endedAt = new Date().toISOString();
let displayStream, microphoneStream, audioContext, audioNodes = [], sockets = {};
let running = false, starting = false, stopping = false, aiSince = 0, aiEpoch = 0;
let interviewerStartedAt = 0, interviewerBuffer = [], questionJobs = Promise.resolve(), answerJobs = Promise.resolve(), activeAnswerController;
let lastPlaybackAt = 0;
let pipWindow = null;

const clock = iso => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
function persist() { sessionStorage.setItem(storeKey, JSON.stringify(session)); }
function status(text, error = false) { ui.status.textContent = text; ui.status.classList.toggle('error', error); }
function buttons() {
  ui.start.disabled = running || starting || stopping;
  ui.end.disabled = !running;
  ui.float.disabled = !running && !session.startedAt;
  ui.export.disabled = stopping || !session.entries.length;
  ui['ai-switch'].disabled = !running;
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
  if (entry.question || entry.answer !== undefined) {
    const box = document.createElement('div'); box.className = 'answer-box';
    const title = document.createElement('b'); title.textContent = entry.question ? `AI 回答 · ${entry.question}` : 'AI 回答';
    const body = document.createElement('div'); body.textContent = entry.answer || '正在生成…';
    box.append(title, body);
    if (entry.sources?.length) {
      const sources = document.createElement('div'); sources.className = 'source-line';
      sources.textContent = `来源：${entry.sources.map(s => s.title || s.source).join('；')}`;
      box.append(sources);
    }
    card.append(box);
  }
  return card;
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

async function classifyAndAnswer(entry, epoch) {
  if (!ui['ai-switch'].checked || epoch !== aiEpoch) return;
  interviewerBuffer.push(entry.text);
  interviewerBuffer = interviewerBuffer.slice(-3);
  let response;
  try {
    response = await fetch('/api/live/question', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: interviewerBuffer.join(' ').slice(-1200) }) });
    if (!response.ok) throw Error('问题判断失败');
    response = await response.json();
  } catch (error) { status(error.message, true); return; }
  if (!ui['ai-switch'].checked || epoch !== aiEpoch || !response.question) return;
  interviewerBuffer = [];
  entry.question = response.question;
  entry.answer = '';
  persist(); renderLog();
  answerJobs = answerJobs.then(() => generateAnswer(entry, epoch));
}

async function generateAnswer(entry, epoch) {
  if (!ui['ai-switch'].checked || epoch !== aiEpoch) return;
  activeAnswerController = new AbortController();
  const controller = activeAnswerController;
  const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch('/api/ask/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: entry.question }), signal: controller.signal });
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
        if (event.type === 'done') { entry.answer = event.result.answer; entry.sources = event.result.sources || []; entry.sourceMode = event.result.source_mode; persist(); renderLog(); return; }
      }
    }
    throw Error('回答连接中断');
  } catch (error) {
    entry.answer = error.name === 'AbortError' ? 'AI 回答已停止或超时。' : `回答失败：${error.message}`;
    persist(); renderLog();
  } finally { clearTimeout(timeout); if (activeAnswerController === controller) activeAnswerController = null; }
}

function onAsrEvent(role, event) {
  if (event.type === 'partial') { ui[`${role}-preview`].textContent = event.text || '识别中…'; return; }
  if (event.type === 'speech_started' && role === 'interviewer') { interviewerStartedAt = Date.now(); return; }
  if (event.type === 'error') { status(`${role === 'me' ? '麦克风' : '面试官'}识别失败：${event.message}`, true); if (running) endSession(); return; }
  if (event.type !== 'transcript') return;
  ui[`${role}-preview`].textContent = event.text || '等待语音…';
  const entry = addTranscript(role, event.text || '');
  if (role === 'interviewer' && entry && ui['ai-switch'].checked && interviewerStartedAt >= aiSince) {
    const epoch = aiEpoch;
    questionJobs = questionJobs.then(() => classifyAndAnswer(entry, epoch)).catch(error => status(`问题处理失败：${error.message}`, true));
  }
  if (role === 'interviewer') interviewerStartedAt = Date.now();
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
    persist(); renderLog(); interviewerBuffer = []; aiSince = 0; aiEpoch++;
    ui['ai-switch'].checked = false;
    interviewerStartedAt = Date.now(); lastPlaybackAt = 0;
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
    status(`正在记录双方语音。麦克风：${micTrack.label || '默认设备'}。AI 回答默认关闭。`);
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
  session.endedAt = new Date().toISOString(); persist(); stopping = false; buttons();
  status('会话已结束。记录可导出；开始新会话会覆盖当前记录。');
}

function exportSession() {
  if (!session.entries.length) return;
  const lines = ['# Atlas 通话记录', '', `开始：${session.startedAt}`, `结束：${session.endedAt || '进行中'}`, ''];
  for (const entry of session.entries) {
    lines.push(`## ${clock(entry.at)} ${entry.role === 'me' ? '我' : '面试官'}`, '', entry.text, '');
    if (entry.question) lines.push(`**识别问题：** ${entry.question}`, '', `**AI 回答：** ${entry.answer || '未完成'}`, '');
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
  if (activeAnswerController) activeAnswerController.abort();
  aiSince = ui['ai-switch'].checked ? Date.now() : 0;
  status(ui['ai-switch'].checked ? 'AI 已开启：只回答现在开始的面试官问题。' : 'AI 已关闭：只记录双方语音。');
};
persist(); renderLog();
