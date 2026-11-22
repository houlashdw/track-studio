# Трек Студия — памятка для нового чата

## Пользователь
- Пишет по-русски, коротко. Отвечать коротко, по делу; результат — mp3 и APK прямо в чат (SendUserFile).
- Оценивает **на слух**, фидбек короткими фразами («хомяк», «скучно», «обрывы»). Сам я слушать не могу → «уши» Gemini (ниже) + замеры.
- Просит экономить лимиты: работать кусками, после куска — отправить результат и остановиться.
- Разрешение: «на все accept», если нужно что-то нажать.

## Безопасность (строго)
- Репо houlashdw/track-studio **публичный**. Ключи (Modal, Kaggle, Gemini, TS_KEY, KEYSTORE_PASS) — только в GitHub Secrets, никогда в чат/код/логи.
- Выходы тестов шифруются `tools/enc_out.sh` публичным ключом `server/out_pub.pem`. Закрытый ключ — только `/root/.ts_out.pem` на машине сессии, не печатать и не коммитить.
- **Новый чат = новый контейнер = закрытого ключа нет.** Сделать новую пару и закоммитить публичный:
  `umask 077; openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out /root/.ts_out.pem; openssl pkey -in /root/.ts_out.pem -pubout -out server/out_pub.pem`
- Не клонировать голоса реальных людей; голос — только сгенерированный ACE.

## Что это
- Android-приложение (WebView `app/index.html`), сборка APK — `android.yml` на push, APK в ветке `apk`: `git show origin/apk:TrackStudio.apk > TrackStudio.apk`.
- Сервер Modal `server/modal_ace.py` (app `track-studio-ace`), URL `https://fantasydrawingart--track-studio-ace-ace-web.us-east.modal.run`. Режим приложения по умолчанию «Reaper + вокалоид» → `POST /track` → опрос `/job/{id}`.
- Конвейер трека (`server/daw/track.py::make_track`, класс `Mixer` в modal_ace):
  1. бит: `server/daw/pack.py::build` — аранжировщик из пака лупов (`/pack/pack1` в volume `ts-ace`) + киты `app/smp`; форма, генератор барабанов по жанру (`gen_drums`), битсвитч, стоп/заикание перед хуком; рендер Reaper headless (`server/reaper/build.lua`);
  2. второй проход: громкость секций по кривой `TGT` (подъём к хуку);
  3. голос: ACE-Step lego (`AceLego`) поёт под бит, текст по секциям (`lyrics_for`), тембр/язык/текст случайны по сиду (`pick_vocal`);
  4. тональность: `chord_fix` (созвучие с аккордами) → **транспонируем бит** под голос (голос не сдвигать — «хомяк»), мягкий автотюн `autotune` (rubberband --pitchmap, задержка калибруется `rb_offset`);
  5. баланс до мастера (`fader`, want −4.5), стак-дубли на хуках, выбросы хвостов в эхо (`voxthrow`), мастер мягкий, громкость −11 LUFS и потолок через ffmpeg.

## Как проверять (одним заходом)
- `bash tools/run_wf.sh deploy` → `bash tools/run_wf.sh tracktest` (3 трека, сиды 101–103, свои — через input ksrc="1 2 3").
- tracktest сам прогоняет «уши»: `tools/listen_track.py` (Gemini: сначала gemini-4-neon, затем gemini-4-hydrogen, затем flash; ключ — секрет GEMINI_API_KEY). Разборы `.md` с оценками JSON (glue, key_match, rhythm, cuts, vocal_life, beat, mix, overall).
- Результаты: `bash tools/dec_out.sh modal-out <папка>` → `track_s*.mp3`, `track_s*.md`, `st/s*_beat.mp3|vox_raw.mp3|vox_fx.mp3|info.json|mix.log`.
- Свежедобавленный workflow сразу после пуша может отдавать 404 на dispatch — повторить через несколько секунд.
- Lua проверять: `pip install lupa` и `load()`; Python — py_compile. Reaper локально не скачать (сайт закрыт прокси).

## Состояние (22.11.2026)
- Лучшее: drill-трек с новыми барабанами — Gemini 8/10. Пользователю: голос «гораздо лучше» после отказа от сдвига тона.
- Следующее: **мелодия** скучная (луп крутится без изменений) → смена лупа по ходу, чопы, контрмелодия, фильтр в интро/брейке. Потом — глайды 808, при желании вокальные эдлибы.
- Gemini Neon требует платного ключа (биллинг в AI Studio); без него работает Hydrogen.
