"""Gemini слушает каждый mp3 из папки и пишет разбор. Промпт — как у звукорежиссёра, сравнение с уровнем Suno."""
import os, sys, glob, json, time
from google import genai
src, dst = sys.argv[1], sys.argv[2]
cl = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
note = os.environ.get('NOTE', '')
ctx = open(os.path.join(src, 'context.txt'), encoding='utf-8').read() if os.path.exists(os.path.join(src, 'context.txt')) else ''
P = """Ты — опытный звукорежиссёр и продюсер, слушаешь трек ушами как человек. Контекст: """ + ctx + """
Ответь по-русски, конкретно и с таймкодами (мм:сс). Разбери:
1) ГОЛОС: искажения (роботизация, металл, хрип, «двойной голос», щелчки, заикание) — где именно; естественность; эмоция; попадание в ноты и ритм; разборчивость слов (перечисли, какие слова не понял).
2) МЕЛОДИЯ И ФЛОУ голоса: интересна ли, повторяется ли, цепляет ли хук, как сидит на бите.
3) БИТ И АРАНЖИРОВКА: какие инструменты слышишь, звучат ли они реально или дёшево, есть ли развитие, переходы, что лишнее, что пусто.
4) СВЕДЕНИЕ: баланс голос/бит, грязь, громкость, низ, верх, пространство.
5) Сравни с типичным треком Suno v4/v5: что хуже, что лучше — по пунктам.
6) ТОП-5 конкретных исправлений по важности (что сделать в обработке/синтезе, чтобы звучало как у профи).
Оценки 1–10: голос, мелодия, бит, сведение, общее. В конце строка JSON: {"voice":..,"melody":..,"beat":..,"mix":..,"overall":..}""" + ("\nОтдельно проверь: " + note if note else '')
for f in sorted(glob.glob(os.path.join(src, '*.mp3'))):
    up = cl.files.upload(file=f)
    while up.state.name == 'PROCESSING': time.sleep(2); up = cl.files.get(name=up.name)
    for model in ['gemini-2.5-pro', 'gemini-2.5-flash']:
        try: r = cl.models.generate_content(model=model, contents=[up, P]); txt = r.text; break
        except Exception as e: txt = 'ошибка ' + model + ': ' + str(e)
    open(os.path.join(dst, os.path.basename(f)[:-4] + '.md'), 'w', encoding='utf-8').write('# ' + os.path.basename(f) + ' (' + model + ')\n\n' + txt)
    print(os.path.basename(f), 'ok')
