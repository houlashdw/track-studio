-- Сборщик бита внутри Reaper: читает проект (Lua-таблица от daw.py), строит дорожки, клипы, MIDI, родные плагины Cockos,
-- посылы на реверб/дилей, сайдчейн 808 от кика, мастер-цепочку; рендерит и выходит.
-- TS_PROJ=<project.lua> TS_LOG=<log> [TS_MGAIN=дБ]  reaper -nosplash -newinst build.lua
local P = dofile(os.getenv('TS_PROJ'))
local logf = io.open(os.getenv('TS_LOG') or '/tmp/build.log', 'w')
local function L(...) local t = {} for i = 1, select('#', ...) do t[#t + 1] = tostring(select(i, ...)) end logf:write(table.concat(t, ' ') .. '\n'); logf:flush() end
local MG = tonumber(os.getenv('TS_MGAIN') or '0')
local function db(x) return 10 ^ (x / 20) end
local seen = {}

-- параметр по имени (часть имени, без регистра)
local function pidx(tr, fx, name)
  local n = reaper.TrackFX_GetNumParams(tr, fx); name = name:lower()
  for i = 0, n - 1 do local _, nm = reaper.TrackFX_GetParamName(tr, fx, i, ''); if nm:lower():find(name, 1, true) then return i end end
  return nil
end
-- выставить параметр так, чтобы его отображаемое значение стало target (бисекция по нормализованному значению)
local function setv(tr, fx, name, target)
  local i = pidx(tr, fx, name); if not i then L('  нет параметра', name); return end
  local function val(x) local _, s = reaper.TrackFX_FormatParamValueNormalized(tr, fx, i, x, ''); local v = tonumber((s:gsub(',', '.')):match('[-+]?%d*%.?%d+')); if s:lower():find('inf') then v = (s:find('-') and -1e9 or 1e9) end; return v, s end
  local a, b = 0.0, 1.0; local va, vb = val(a), val(b)
  if not va or not vb then L('  не число', name); return end
  local up = vb > va
  for _ = 1, 40 do local m = (a + b) / 2; local vm = val(m); if not vm then break end; if (vm < target) == up then a = m else b = m end end
  reaper.TrackFX_SetParamNormalized(tr, fx, i, (a + b) / 2)
  local _, s = reaper.TrackFX_GetFormattedParamValue(tr, fx, i, ''); L('  ', name, '=', s)
end
-- перечисляемый параметр: выбрать вариант, в названии которого есть text
local function setenum(tr, fx, name, text)
  local i = pidx(tr, fx, name); if not i then L('  нет параметра', name); return end
  for k = 0, 100 do local x = k / 100; local _, s = reaper.TrackFX_FormatParamValueNormalized(tr, fx, i, x, ''); if s:lower():find(text:lower(), 1, true) then reaper.TrackFX_SetParamNormalized(tr, fx, i, x); L('  ', name, '=', s); return end end
  local opts = {} for k = 0, 20 do local _, s = reaper.TrackFX_FormatParamValueNormalized(tr, fx, i, k / 20, ''); opts[#opts + 1] = s end
  L('  вариант не найден', name, text, 'есть:', table.concat(opts, ' / '))
end
local function addfx(tr, nm)
  local fx = reaper.TrackFX_AddByName(tr, nm, false, -1)
  if fx < 0 then L('НЕТ ПЛАГИНА', nm); return nil end
  if not seen[nm] then seen[nm] = true; local n = reaper.TrackFX_GetNumParams(tr, fx); local t = {}
    for i = 0, math.min(n, 40) - 1 do local _, p = reaper.TrackFX_GetParamName(tr, fx, i, ''); t[#t + 1] = p end; L('параметры', nm, ':', table.concat(t, ' | ')) end
  return fx
end
local function eq(tr, bands)   -- bands: {{тип, частота, дБ, ширина}}; ReaEQ по умолчанию: 4 полосы (low shelf, band, band, high shelf)
  local fx = addfx(tr, 'ReaEQ (Cockos)'); if not fx then return end
  local SUF = {'low shelf', 'band 2', 'band 3', 'high shelf 4'}
  for i, b in ipairs(bands) do
    setv(tr, fx, 'freq-' .. SUF[i], b[2]); setv(tr, fx, 'gain-' .. SUF[i], b[3]); if b[4] then setv(tr, fx, 'bw-' .. SUF[i], b[4]) end
  end
end
local function comp(tr, th, ratio, att, rel, aux)
  local fx = addfx(tr, 'ReaComp (Cockos)'); if not fx then return end
  setv(tr, fx, 'Threshold', th); setv(tr, fx, 'Ratio', ratio); setv(tr, fx, 'Attack', att); setv(tr, fx, 'Release', rel)
  if aux then setenum(tr, fx, 'SignIn', 'aux'); setenum(tr, fx, 'AudIn', 'main') end
  return fx
end

reaper.Main_OnCommand(40859, 0)
reaper.SetCurrentBPM(0, P.bpm, false)
local tracks, idx = {}, 0
local function newtrack(name)
  reaper.InsertTrackAtIndex(idx, true); local tr = reaper.GetTrack(0, idx); idx = idx + 1
  reaper.GetSetMediaTrackInfo_String(tr, 'P_NAME', name, true); return tr
end

-- шины: реверб и дилей
local rev = newtrack('REVERB'); local fxr = addfx(rev, 'ReaVerbate (Cockos)'); if fxr then setv(rev, fxr, 'Room size', 70); setv(rev, fxr, 'Dampening', 50); setv(rev, fxr, 'Wet', 0); setv(rev, fxr, 'Dry', -150) end
eq(rev, {{0, 300, -6}, {0, 1000, 0}, {0, 4000, 0}, {0, 9000, -4}})
local dly = newtrack('DELAY'); local fxd = addfx(dly, 'ReaDelay (Cockos)'); if fxd then setv(dly, fxd, 'Wet', 0); setv(dly, fxd, 'Dry', -150) end

for _, t in ipairs(P.tracks) do
  local tr = newtrack(t.name); tracks[t.name] = tr
  reaper.SetMediaTrackInfo_Value(tr, 'D_VOL', db(t.vol or 0)); reaper.SetMediaTrackInfo_Value(tr, 'D_PAN', t.pan or 0)
  if t.kind == 'audio' then
    local src = reaper.PCM_Source_CreateFromFile(t.file)
    for _, it in ipairs(t.items) do
      local item = reaper.AddMediaItemToTrack(tr); reaper.SetMediaItemInfo_Value(item, 'D_POSITION', it.t); reaper.SetMediaItemInfo_Value(item, 'D_LENGTH', it.d)
      reaper.SetMediaItemInfo_Value(item, 'D_FADEOUTLEN', it.fo or 0.015); reaper.SetMediaItemInfo_Value(item, 'D_FADEINLEN', it.fi or (t.name == '808' and 0.025 or 0.001))   -- 808 мягко входит после щелчка кика (вместо сайдчейна)
      if it.loop then reaper.SetMediaItemInfo_Value(item, 'B_LOOPSRC', 1) else reaper.SetMediaItemInfo_Value(item, 'B_LOOPSRC', 0) end
      local tk = reaper.AddTakeToMediaItem(item); reaper.SetMediaItemTake_Source(tk, src)
      reaper.SetMediaItemTakeInfo_Value(tk, 'D_PLAYRATE', it.rate); reaper.SetMediaItemTakeInfo_Value(tk, 'B_PPITCH', it.pp and 1 or 0); reaper.SetMediaItemTakeInfo_Value(tk, 'D_VOL', it.v * db(t.ig or 0))
      if it.off then reaper.SetMediaItemTakeInfo_Value(tk, 'D_STARTOFFS', it.off) end
      if it.pitch then reaper.SetMediaItemTakeInfo_Value(tk, 'D_PITCH', it.pitch) end
    end
  else
    local item = reaper.CreateNewMIDIItemInProj(tr, 0, P.total, false); local tk = reaper.GetActiveTake(item)
    for _, n in ipairs(t.notes) do
      local a = reaper.MIDI_GetPPQPosFromProjTime(tk, n.t); local b = reaper.MIDI_GetPPQPosFromProjTime(tk, n.t + n.d)
      reaper.MIDI_InsertNote(tk, false, false, a, b, 0, n.m, math.min(127, n.v), true)
    end
    reaper.MIDI_Sort(tk)
    if t.inst == 'synth' then
      local fx = addfx(tr, 'ReaSynth (Cockos)'); if fx then setv(tr, fx, 'Saw mix', -6); setv(tr, fx, 'Attack', 5); setv(tr, fx, 'Release', 120) end
      local ds = addfx(tr, 'JS: Distortion'); if not ds then addfx(tr, 'JS: Saturation') end
    else
      for _, z in ipairs(t.zones) do   -- мульти-сэмпл: по RS5K на зону клавиш
        local fx = addfx(tr, 'ReaSamplOmatic5000 (Cockos)')
        if fx then
          reaper.TrackFX_SetNamedConfigParm(tr, fx, 'FILE0', z.file); reaper.TrackFX_SetNamedConfigParm(tr, fx, 'DONE', '')
          setv(tr, fx, 'Note range start', z.lo); setv(tr, fx, 'Note range end', z.hi)   -- высота идёт по клавишам: начало зоны → (lo−корень), конец → (hi−корень)
          setv(tr, fx, 'Pitch for start note', z.lo - z.root); setv(tr, fx, 'Pitch for end note', z.hi - z.root); setv(tr, fx, 'Obey note-offs', 1); setv(tr, fx, 'Release', 250)
          setv(tr, fx, 'Volume', t.ig or 0)
        end
      end
    end
  end
  -- обработка дорожки: EQ/компрессор/сатурация по роли
  local nm = t.name
  if nm == 'kick' then eq(tr, {{0, 55, 2}, {0, 300, -3, 1.5}, {0, 3500, 2, 1}, {0, 10000, 0}}); comp(tr, -14, 3, 3, 60)
  elseif nm == 'snare' or nm == 'clap' then eq(tr, {{0, 150, -4}, {0, 900, -2, 1.5}, {0, 5000, 2, 1}, {0, 10000, 2}}); comp(tr, -16, 3, 5, 80)
  elseif nm == 'hat' or nm == 'ohat' then eq(tr, {{0, 400, -12}, {0, 2000, 0}, {0, 8000, 1}, {0, 12000, 2}})
  elseif nm == '808' then
    eq(tr, {{0, 35, 2}, {0, 250, -3, 1.5}, {0, 1200, 1, 1}, {0, 6000, -6}})
    local sat = addfx(tr, 'JS: Saturation'); if sat then setv(tr, sat, 'Amount', P.drive * 100) end
    comp(tr, -20, 4, 1, 120)
  elseif nm == 'mel' then   -- петля мелодии из пака: низ освобождаем под 808, чуть плотнее
    eq(tr, {{0, 140, -10}, {0, 350, -2, 1.5}, {0, 3000, 0.5, 1}, {0, 10000, 0}}); comp(tr, -20, 2, 15, 150)
  elseif nm == 'drumloop' then   -- живые барабаны: панч и воздух, чуть склеить
    eq(tr, {{0, 60, 1.5}, {0, 400, -2, 1.5}, {0, 4000, 1, 1}, {0, 11000, 1.5}}); comp(tr, -16, 2.5, 8, 90)
  elseif nm == 'bassloop' then
    eq(tr, {{0, 35, 1}, {0, 250, -2, 1.5}, {0, 1200, 0, 1}, {0, 5000, -6}})
    local sat = addfx(tr, 'JS: Saturation'); if sat then setv(tr, sat, 'Amount', P.drive * 60) end
    comp(tr, -20, 3, 3, 120)
  elseif nm == 'melody' or nm == 'pad' then
    local bright = (t.inst == 'glock' or t.inst == 'chimes' or t.inst == 'cowbell') and -7 or -2   -- колокола очень звонкие: верх приглушаем
    eq(tr, {{0, 180, -8}, {0, 600, -2, 1.5}, {0, 3000, bright / 3, 1}, {0, 6500, bright}}); comp(tr, -18, 2.5, 10, 150)
    if nm == 'pad' then local ch = addfx(tr, 'JS: Chorus'); end
  end
  -- посылы
  local function send(dst, dbv) local s = reaper.CreateTrackSend(tr, dst); reaper.SetTrackSendInfo_Value(tr, 0, s, 'D_VOL', db(dbv)) end
  if nm == 'mel' then send(rev, -16) elseif nm == 'melody' then send(rev, -10); send(dly, -14) elseif nm == 'pad' then send(rev, -8) elseif nm == 'clap' or nm == 'snare' then send(rev, -18) end
end
if fxd then local q = 60 / P.bpm; setv(dly, fxd, '1: Length (time)', q * 750); setv(dly, fxd, '1: Feedback', -12) end

-- мастер: EQ → многополосный ReaXcomp → склейка ReaComp → ReaLimit
local M = reaper.GetMasterTrack(0)
reaper.SetMediaTrackInfo_Value(M, 'D_VOL', 1)
eq(M, {{0, 30, -6}, {0, 250, -1, 2}, {0, 3000, 0.5, 2}, {0, 12000, 1.5}})
addfx(M, 'ReaXcomp (Cockos)')
local gl = comp(M, -10, 2, 10, 200)
local lim = addfx(M, 'ReaLimit (Cockos)'); if lim then setv(M, lim, 'Threshold', P.loud + MG); setv(M, lim, 'Ceiling', -1) end

-- рендер
reaper.GetSetProjectInfo(0, 'RENDER_SETTINGS', 0, true); reaper.GetSetProjectInfo(0, 'RENDER_BOUNDSFLAG', 0, true)
reaper.GetSetProjectInfo(0, 'RENDER_STARTPOS', 0, true); reaper.GetSetProjectInfo(0, 'RENDER_ENDPOS', P.total, true)
reaper.GetSetProjectInfo(0, 'RENDER_SRATE', 44100, true); reaper.GetSetProjectInfo(0, 'RENDER_CHANNELS', 2, true)
local dir, fn = P.out:match('^(.*)/([^/]+)%.wav$')
reaper.GetSetProjectInfo_String(0, 'RENDER_FILE', dir, true); reaper.GetSetProjectInfo_String(0, 'RENDER_PATTERN', fn, true)
reaper.GetSetProjectInfo_String(0, 'RENDER_FORMAT', 'ZXZhdxgAAQ==', true)
reaper.Main_SaveProjectEx(0, dir .. '/' .. fn .. '.rpp', 0)
L('рендер…'); reaper.Main_OnCommand(42230, 0); L('РЕНДЕР ГОТОВ')
reaper.Main_OnCommand(40004, 0)
