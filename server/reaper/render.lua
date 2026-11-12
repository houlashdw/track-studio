-- Reaper без экрана: дорожки из папки IN → у каждой своя цепочка плагинов (FXMAP) → рендер в OUT/mix.wav, потом выход
local IN, OUT = os.getenv('TS_IN') or '/tmp/rin', os.getenv('TS_OUT') or '/tmp/rout'
local FXMAP = os.getenv('TS_FX') or ''   -- "drums=JS: ...|ReaComp;bass=...;*=..."
local log = io.open(OUT .. '/log.txt', 'w')
local function L(s) log:write(s .. '\n'); log:flush() end
local chains = {}
for part in string.gmatch(FXMAP, '([^;]+)') do local k, v = part:match('^([^=]+)=(.*)$'); if k then chains[k] = v end end
reaper.Main_OnCommand(40859, 0) -- новый проект
local i = 0
local p = io.popen('ls "' .. IN .. '"')
for f in p:lines() do
  if f:match('%.wav$') or f:match('%.mp3$') then
    local name = f:gsub('%.%w+$', '')
    reaper.InsertTrackAtIndex(i, true); local tr = reaper.GetTrack(0, i); reaper.GetSetMediaTrackInfo_String(tr, 'P_NAME', name, true)
    reaper.SetOnlyTrackSelected(tr); reaper.SetEditCurPos(0, false, false)
    reaper.InsertMedia(IN .. '/' .. f, 0)
    local ch = chains[name] or chains['*'] or ''
    for fx in string.gmatch(ch, '([^|]+)') do local r = reaper.TrackFX_AddByName(tr, fx, false, -1); L(name .. ' + ' .. fx .. ' -> ' .. r) end
    i = i + 1
  end
end
p:close()
-- рендер всего мастера: 44.1 кГц, wav
reaper.GetSetProjectInfo(0, 'RENDER_SETTINGS', 0, true)          -- мастер
reaper.GetSetProjectInfo(0, 'RENDER_BOUNDSFLAG', 1, true)        -- весь проект
reaper.GetSetProjectInfo(0, 'RENDER_SRATE', 44100, true)
reaper.GetSetProjectInfo(0, 'RENDER_CHANNELS', 2, true)
reaper.GetSetProjectInfo_String(0, 'RENDER_FILE', OUT, true)
reaper.GetSetProjectInfo_String(0, 'RENDER_PATTERN', 'mix', true)
reaper.GetSetProjectInfo_String(0, 'RENDER_FORMAT', 'ZXZhdxgAAQ==', true) -- wav 24 бит
L('дорожек: ' .. i)
reaper.Main_OnCommand(42230, 0) -- рендер с текущими настройками, закрыть окно
L('рендер готов')
reaper.Main_OnCommand(40004, 0) -- выход
