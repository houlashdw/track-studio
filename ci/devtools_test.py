"""Управляет студией внутри APK через отладчик WebView и пишет ci-out/result.json."""
import json, os, subprocess, time, urllib.request
import websocket

OUT = 'ci-out'
R = {'steps': []}
EVENTS = []


def save():
    with open(OUT + '/result.json', 'w', encoding='utf-8') as f:
        json.dump(R, f, ensure_ascii=False, indent=1)


def shot(name):
    png = subprocess.run(['adb', 'exec-out', 'screencap', '-p'], capture_output=True).stdout
    with open('%s/%s.png' % (OUT, name), 'wb') as f:
        f.write(png)


def step(name, value):
    print(name, json.dumps(value, ensure_ascii=False)[:600], flush=True)
    R['steps'].append({'name': name, 'value': value, 't': round(time.time() - T0, 1)})
    save()


T0 = time.time()
page = None
for _ in range(90):
    try:
        pages = json.load(urllib.request.urlopen('http://127.0.0.1:9222/json', timeout=5))
        page = next((p for p in pages if 'appassets' in p.get('url', '')), None)
        if page:
            break
    except Exception as e:
        R['connect_err'] = str(e)
    time.sleep(2)
if not page:
    step('connect', {'ok': False})
    raise SystemExit(1)
ws = websocket.create_connection(page['webSocketDebuggerUrl'], timeout=120, suppress_origin=True)
MID = [0]


def call(method, params=None):
    MID[0] += 1
    mid = MID[0]
    ws.send(json.dumps({'id': mid, 'method': method, 'params': params or {}}))
    while True:
        m = json.loads(ws.recv())
        if m.get('id') == mid:
            return m
        if m.get('method') in ('Runtime.exceptionThrown', 'Runtime.consoleAPICalled', 'Log.entryAdded'):
            p = m.get('params', {})
            if m['method'] == 'Runtime.consoleAPICalled':
                txt = p.get('type', '') + ': ' + ' '.join(str(x.get('value', x.get('description', ''))) for x in p.get('args', []))[:400]
            elif m['method'] == 'Log.entryAdded':
                e = p.get('entry', {})
                txt = 'log ' + e.get('level', '') + ': ' + (e.get('text') or '')[:300] + ' ' + (e.get('url') or '')[:120]
            else:
                d = p.get('exceptionDetails', {})
                txt = 'EXCEPTION: ' + (d.get('text') or '') + ' ' + str((d.get('exception') or {}).get('description'))[:400]
            EVENTS.append('%.0f %s' % (time.time() - T0, txt))
            R['console'] = EVENTS[-200:]


def ev(expr):
    m = call('Runtime.evaluate', {'expression': expr, 'awaitPromise': True, 'returnByValue': True})
    r = m.get('result', {})
    if 'exceptionDetails' in r:
        d = r['exceptionDetails']
        return {'__err': (d.get('text') or '') + ' ' + str((d.get('exception') or {}).get('description'))[:400]}
    return (r.get('result') or {}).get('value')


STATUS = "(document.querySelector('#statusText').title || document.querySelector('#statusText').textContent)"


def start_build(setup_js):
    """Настроить поля, дождаться свободной кнопки «Создать» и нажать её; вернуть True, если сборка пошла."""
    ev(setup_js)
    for attempt in range(6):
        wait("!S.busy && !document.querySelector('#make').disabled", 600, 2)
        time.sleep(3)   # смена режима может запустить пересборку — даём ей начаться
        if ev("!S.busy && !document.querySelector('#make').disabled") is not True:
            continue
        ev("(()=>{ document.querySelector('#player').dataset.ready=''; document.querySelector('#make').click(); return 1; })()")
        if wait("S.busy === true || document.querySelector('#player').dataset.ready === '1'", 20, 1) is not None:
            return attempt + 1
    return 0


def snap():
    return ev("""({status: %s, busy: !!S.busy, ready: document.querySelector('#player').dataset.ready, line: (document.querySelector('#voiceLine')||{}).textContent,
      kv2: (()=>{ const k=globalThis.__kv2||{}; return {mode:k.mode, sing:k.sing, busy:!!k.busy, q:(k.q||[]).length, genN:k.genN, genMs:k.genMs|0}; })(),
      mem: performance.memory ? Math.round(performance.memory.usedJSHeapSize/1048576) : null})""" % STATUS)


def wait(expr, limit, every=5):
    t = time.time()
    while time.time() - t < limit:
        v = ev(expr)
        if v is True:
            return round(time.time() - t, 1)
        time.sleep(every)
    return None


call('Runtime.enable')
call('Log.enable')
step('env', ev("""({href: location.href, secure: isSecureContext, caches: typeof caches, idb: typeof indexedDB, ua: navigator.userAgent,
  bridge: typeof TSAndroid, platform: (window.TSAndroid && TSAndroid.platform()), sw: !!(navigator.serviceWorker && navigator.serviceWorker.controller),
  title: document.title, analyze: typeof analyze, installHidden: document.querySelector('#installBtn').hidden, inApp: document.documentElement.classList.contains('in-app'),
  w: innerWidth, h: innerHeight, dpr: devicePixelRatio})"""))

ev("""(()=>{ const p=document.querySelector('#prompt'); p.value='хардстайл 150 bpm'; p.dispatchEvent(new Event('input')); return 1; })()""")
time.sleep(1)
step('understood', ev("document.querySelector('#understood').textContent"))

# 1. бит без голоса
started = start_build("""(()=>{ document.querySelector('#modeSeg button[data-v="off"]').click();
  const p=document.querySelector('#prompt'); p.value='jumpstyle 145 bpm, энергично'; p.dispatchEvent(new Event('input')); return 1; })()""")
step('beat_started', started)
took = wait("document.querySelector('#player').dataset.ready==='1' && !S.busy", 900)
step('beat', {'seconds': took, 'status': ev("document.querySelector('#statusText').title || document.querySelector('#statusText').textContent"),
              'dur': ev("S.buffer ? +S.buffer.duration.toFixed(1) : null")})
shot('2_beat')

# 2. сохранение: обычная ссылка download и кнопка WAV
step('save_link', ev("""(async()=>{ const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob(['проверка'],{type:'text/plain'})); a.download='ci-test.txt'; document.body.append(a); a.click(); a.remove(); await new Promise(r=>setTimeout(r,2500)); return document.querySelector('#toast') ? document.querySelector('#toast').textContent : ''; })()"""))
step('save_wav', ev("""(async()=>{ const b=document.querySelector('#dlWav'); if(!b||b.disabled) return 'нет кнопки'; b.click(); await new Promise(r=>setTimeout(r,8000)); return document.querySelector('#toast') ? document.querySelector('#toast').textContent : ''; })()"""))
step('downloads_after_beat', subprocess.run(['adb', 'shell', 'ls', '-la', '/sdcard/Download/TrackStudio'], capture_output=True, text=True).stdout)

# 3. треки с голосом (модель голоса качается при первом запуске): рэп и пение
LY = ("[Verse]\\nCity lights are calling out my name tonight\\nHold me close and never let me go\\n"
      "We don't stop, we just turn it up louder\\nEvery heartbeat matching with the drum\\n"
      "[Chorus]\\nStars are falling, I will sing tonight\\nThis is our song, so sing it loud")
TRACKS = [('rap', 'хардстайл 150 bpm', 'voice'), ('sing', 'аниме-опенинг, энергично, милый голос, 150 bpm', 'voice_sing')]


def voice_track(mode, prompt, name):
    t0 = time.time()
    started = start_build("""(()=>{ document.querySelector('#modeSeg button[data-v="%s"]').click();
      const l=document.querySelector('#lyrics'); l.value="%s"; l.dispatchEvent(new Event('input'));
      const p=document.querySelector('#prompt'); p.value='%s'; p.dispatchEvent(new Event('input')); return 1; })()""" % (mode, LY, prompt))
    step(name + '_started', {'attempt': started, 'after_s': round(time.time() - t0, 1), 'snap': snap()})
    took = None
    snaps = R.setdefault(name + '_snaps', [])
    last = 0
    while time.time() - t0 < 1800:
        if ev("document.querySelector('#player').dataset.ready==='1' && !S.busy") is True:
            took = round(time.time() - t0, 1)
            break
        if time.time() - last > 60:
            last = time.time()
            snaps.append([round(time.time() - t0), snap()])
            save()
        time.sleep(5)
    step(name, {'seconds': took, 'status': ev(STATUS), 'voxInfo': ev("S.voxInfo || null"), 'voxFail': ev("S.voxFail || null"),
                'kv2': ev("JSON.stringify(globalThis.__kv2 || {}).slice(0,400)"), 'dur': ev("S.buffer ? +S.buffer.duration.toFixed(1) : null"),
                'lineDiag': ev("JSON.stringify(S.lineDiag || null)")})
    shot('3_' + name)
    if took:
        step(name + '_report', ev("""(async()=>{ try { const z=await makeReport(); return await window.__tsSend(z,'report-%s.zip',false); } catch(e){ return 'ERR '+e.message; } })()""" % name))


if os.environ.get('VOICE', 'true') == 'true':
    for mode, prompt, name in TRACKS:
        voice_track(mode, prompt, name)

# 4. певческая модель (DiffSinger, нативно): скачать и спеть
if os.environ.get('VOICE', 'true') == 'true':
    ev("(()=>{ TSAndroid.dsInstall(); return 1; })()")
    t0 = time.time(); st = None
    while time.time() - t0 < 1200:
        st = ev("TSAndroid.dsStatus()")
        if st and ('installed' in st or 'error' in st):
            break
        time.sleep(10)
    step('ds_install', {'status': st, 'seconds': round(time.time() - t0)})
    if st and 'installed' in st:
        voice_track('sing', 'аниме-опенинг, энергично, милый голос, 150 bpm', 'voice_ds')
        step('ds_used', ev("JSON.stringify({used: (globalThis.DSX||{}).used, fail: (globalThis.DSX||{}).fail})"))

step('downloads', subprocess.run(['adb', 'shell', 'ls', '-la', '/sdcard/Download/TrackStudio'], capture_output=True, text=True).stdout)
R['total_s'] = round(time.time() - T0, 1)
save()
