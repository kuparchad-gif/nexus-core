(() => {
  const cfg = window.HERMES_CONFIG || { API_BASE: '', GUEST_ENABLED: true };
  const els = {
    health: document.getElementById('health-pill'),
    modelSelect: document.getElementById('modelSelect'),
    thread: document.getElementById('thread'),
    prompt: document.getElementById('prompt'),
    sendBtn: document.getElementById('sendBtn'),
    loginBtn: document.getElementById('loginBtn'),
    userBadge: document.getElementById('userBadge'),
    logoutBtn: document.getElementById('logoutBtn'),
    statusPanel: document.getElementById('statusPanel'),
    sessionInfo: document.getElementById('sessionInfo'),
    loginModal: document.getElementById('loginModal'),
    uName: document.getElementById('uName'),
    uPass: document.getElementById('uPass'),
    loginGo: document.getElementById('loginGo'),
    guestBtn: document.getElementById('guestBtn'),
    deployBtn: document.getElementById('deployBtn'),
    deployModal: document.getElementById('deployModal'),
    platformSelect: document.getElementById('platformSelect'),
    deployGo: document.getElementById('deployGo'),
    deployClose: document.getElementById('deployClose'),
    deployLog: document.getElementById('deployLog'),
    toasts: document.getElementById('toasts')
  };

  function toast(msg, good=false){
    const t = document.createElement('div');
    t.className = 'toast ' + (good ? 'good':'bad');
    t.textContent = msg;
    els.toasts.appendChild(t);
    setTimeout(()=>{ t.remove(); }, 3500);
  }

  function token(){ return localStorage.getItem('jwt') || ''; }
  function role(){ return localStorage.getItem('role') || 'guest'; }
  function setSession(role, username){
    localStorage.setItem('role', role || 'guest');
    if (username) localStorage.setItem('user', username);
    const badge = role === 'admin' ? 'admin' : 'guest';
    els.userBadge.textContent = badge;
    els.loginBtn.style.display = 'none';
    els.logoutBtn.style.display = 'inline-block';
    els.sessionInfo.textContent = `role=${badge} user=${localStorage.getItem('user')||'-'}`;
    els.deployBtn.style.display = (badge==='admin') ? 'inline-block' : 'none';
  }
  function clearSession(){
    localStorage.removeItem('jwt');
    localStorage.removeItem('role');
    localStorage.removeItem('user');
    els.userBadge.textContent = '';
    els.loginBtn.style.display = 'inline-block';
    els.logoutBtn.style.display = 'inline-block';
    els.sessionInfo.textContent = '-';
    els.deployBtn.style.display = 'none';
  }

  function renderMsg(who, text){
    const d = document.createElement('div');
    d.className = 'msg ' + who;
    d.textContent = text;
    els.thread.appendChild(d);
    els.thread.scrollTop = els.thread.scrollHeight;
  }

  async function api(path, opt={}){
    const headers = opt.headers || {};
    if (token()) headers['Authorization'] = 'Bearer ' + token();
    headers['Content-Type'] = 'application/json';
    const res = await fetch(cfg.API_BASE + path, { ...opt, headers });
    if (!res.ok){
      let detail='Request failed';
      try { const j = await res.json(); detail = j.error || JSON.stringify(j); } catch{}
      throw new Error(detail);
    }
    const ct = res.headers.get('content-type')||'';
    if (ct.includes('application/json')) return res.json();
    return res.text();
  }

  async function refreshHealth(){
    try {
      const h = await api('/health');
      els.health.textContent = 'healthy';
      els.health.style.color = '#1f8b4c';
      els.statusPanel.textContent = JSON.stringify(h,null,2);
    } catch(e) {
      els.health.textContent = 'unavailable';
      els.health.style.color = '#b00020';
      els.statusPanel.textContent = String(e);
    }
  }

  async function refreshModels(){
    try {
      const m = await api('/v1/models');
      const list = (m.models || []).map(x => x.id || x.name || x.model || 'model');
      els.modelSelect.innerHTML = '';
      list.forEach(id => {
        const o = document.createElement('option');
        o.value = id; o.textContent = id;
        els.modelSelect.appendChild(o);
      });
    } catch(e){ toast('Models: '+e.message); }
  }

  async function doChat(){
    const content = els.prompt.value.trim();
    if (!content) return;
    renderMsg('user', content);
    els.prompt.value = '';
    try {
      const model = els.modelSelect.value || undefined;
      const body = { 
        model,
        messages:[{role:'user', content}],
        max_tokens: 256, temperature: 0.3
      };
      const r = await api('/v1/chat/completions', { method:'POST', body: JSON.stringify(body)});
      const text = r.choices?.[0]?.message?.content || JSON.stringify(r);
      renderMsg('assistant', text);
    } catch(e){ toast('Chat failed: '+e.message); }
  }

  // Auth flows
  async function openLogin(){ els.loginModal.classList.remove('hidden'); }
  function closeLogin(){ els.loginModal.classList.add('hidden'); }

  async function signIn(){
    try{
      const resp = await api('/auth/login', { 
        method:'POST',
        body: JSON.stringify({ username: els.uName.value.trim(), password: els.uPass.value })
      });
      localStorage.setItem('jwt', resp.access_token);
      setSession('admin', resp.username || els.uName.value.trim());
      closeLogin();
      toast('Welcome back, '+(resp.username||'admin')+'!', true);
    } catch(e){ toast('Login failed: '+e.message); }
  }

  async function guest(){
    try{
      const r = await api('/auth/guest', { method:'POST' });
      localStorage.setItem('jwt', r.access_token);
      setSession('guest', 'guest');
      closeLogin();
      toast('Guest session ready', true);
    } catch(e){
      toast('Guest failed: '+e.message);
    }
  }

  // Deploy (admin)
  function openDeploy(){
    if (role()!=='admin'){ toast('Admin only'); return; }
    els.deployModal.classList.remove('hidden');
    els.deployLog.textContent = '';
  }
  function closeDeploy(){ els.deployModal.classList.add('hidden'); }

  async function runDeploy(){
    if (role()!=='admin'){ toast('Admin only'); return; }
    els.deployLog.textContent += '⏳ Deploying…\n';
    try {
      const r = await api('/deploy', { method:'POST' });
      els.deployLog.textContent += '✅ ' + JSON.stringify(r) + '\n';
      toast('Deploy done', true);
    } catch(e){
      els.deployLog.textContent += '❌ ' + e.message + '\n';
      toast('Deploy failed: '+e.message);
    }
  }

  // Wire events
  els.sendBtn.onclick = doChat;
  els.prompt.addEventListener('keydown', (e)=>{
    if (e.key==='Enter' && !e.shiftKey){ e.preventDefault(); doChat(); }
  });
  els.loginBtn.onclick = openLogin;
  els.loginGo.onclick = signIn;
  els.guestBtn.onclick = guest;
  els.logoutBtn.onclick = () => { localStorage.removeItem('jwt'); clearSession(); toast('Logged out', true); };
  els.deployBtn.onclick = openDeploy;
  els.deployClose.onclick = closeDeploy;
  els.deployGo.onclick = runDeploy;

  // Boot
  (async () => {
    if (token()){
      setSession(role(), localStorage.getItem('user')||undefined);
    } else {
      clearSession();
      // auto-open login if no token
      openLogin();
    }
    await refreshHealth();
    await refreshModels();
    setInterval(refreshHealth, 5000);
  })();
})();
