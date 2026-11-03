"""Renri RVC: подготовка данных, признаки, обучение, экспорт ONNX и пак для приложения.
Запуск из корня RVC 2.2 (тег 2.2.231006):  python renri_rvc.py <stage> ...
stage: prep <dataset_dir> | f0 | feat | train <epochs> <batch> | export | pack <out.zip>
Без fairseq/parselmouth/pyworld: признаки — ContentVec через transformers, высота — RMVPE (запасной вариант — автокорреляция)."""
import os, sys, json, glob, math, shutil, subprocess, zipfile
import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.getcwd())
EXP = os.environ.get('EXP', 'renri')
ROOT = os.getcwd()
LOG = os.path.join(ROOT, 'logs', EXP)
CK = os.environ.get('CKPT_DIR', LOG)   # куда сохранять прогресс (Google Диск): контрольные точки и готовые веса
SR = 40000
CV_NAME = os.environ.get('CONTENTVEC', 'lengyue233/content-vec-best')


def resample(x, sr0, sr1):
    if sr0 == sr1:
        return x.astype(np.float32)
    g = math.gcd(sr0, sr1)
    return signal.resample_poly(x, sr1 // g, sr0 // g).astype(np.float32)


def stage_prep(ds_dir):
    """как preprocess.py RVC: фильтр 48 Гц, куски по 3.7 с с перекрытием 0.3 с, нормализация 0.9·0.75 + 0.25·исходный"""
    d0, d1 = os.path.join(LOG, '0_gt_wavs'), os.path.join(LOG, '1_16k_wavs')
    if os.path.exists(os.path.join(LOG, 'prep.ok')):
        print('prep: уже сделано (с Диска)'); return
    os.makedirs(d0, exist_ok=True); os.makedirs(d1, exist_ok=True)
    bh, ah = signal.butter(N=5, Wn=48, btype='high', fs=SR)
    files = sorted(glob.glob(os.path.join(ds_dir, '**', '*.*'), recursive=True))
    files = [f for f in files if f.lower().endswith(('.wav', '.flac', '.ogg', '.mp3'))]
    per, ov = 3.7, 0.3; n = 0
    for fi, f in enumerate(files):
        x, sr = sf.read(f, dtype='float32', always_2d=True); x = x.mean(1)
        x = resample(x, sr, SR); x = signal.lfilter(bh, ah, x).astype(np.float32)
        L, step = int(per * SR), int((per - ov) * SR)
        for j, a in enumerate(range(0, max(1, len(x) - int(ov * SR)), step)):
            seg = x[a:a + L]
            if len(seg) < SR * 0.5:
                continue
            mx = np.abs(seg).max()
            if mx < 1e-3 or mx > 2.5:
                continue
            seg = (seg / mx * (0.9 * 0.75)) + (1 - 0.75) * seg
            name = '%d_%d' % (fi, j)
            sf.write(os.path.join(d0, name + '.wav'), seg, SR, subtype='FLOAT')
            sf.write(os.path.join(d1, name + '.wav'), resample(seg, SR, 16000), 16000, subtype='FLOAT')
            n += 1
    open(os.path.join(LOG, 'prep.ok'), 'w').write(str(n))
    print('prep: файлов', len(files), 'кусков', n)


F0_MIN, F0_MAX = 50.0, 1100.0
MEL_MIN, MEL_MAX = 1127 * np.log(1 + F0_MIN / 700), 1127 * np.log(1 + F0_MAX / 700)


def coarse(f0):
    m = 1127 * np.log(1 + f0 / 700)
    m[m > 0] = (m[m > 0] - MEL_MIN) * 254 / (MEL_MAX - MEL_MIN) + 1
    m[m <= 1] = 1; m[m > 255] = 255
    return np.rint(m).astype(np.int64)


def f0_autocorr(x, sr=16000, hop=160):
    W = 1024; out = []
    for a in range(0, len(x), hop):
        s = x[a:a + W]
        if len(s) < W:
            s = np.pad(s, (0, W - len(s)))
        s = s - s.mean(); e = (s ** 2).sum()
        if e < W * 1e-5:
            out.append(0.0); continue
        c = np.fft.irfft(np.abs(np.fft.rfft(s, 2 * W)) ** 2)[:W]; c /= c[0] + 1e-9
        lo, hi = int(sr / 1000), int(sr / 70); k = lo + int(np.argmax(c[lo:hi]))
        out.append(sr / k if c[k] > 0.5 else 0.0)
    return np.array(out, np.float32)


def stage_f0():
    d1 = os.path.join(LOG, '1_16k_wavs'); da, db = os.path.join(LOG, '2a_f0'), os.path.join(LOG, '2b-f0nsf')
    os.makedirs(da, exist_ok=True); os.makedirs(db, exist_ok=True)
    rm = None; p = os.path.join(ROOT, 'assets', 'rmvpe', 'rmvpe.pt')
    if os.path.exists(p):
        import torch
        from infer.lib.rmvpe import RMVPE
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        rm = RMVPE(p, is_half=False, device=dev)
    print('f0:', 'RMVPE' if rm else 'автокорреляция (запасной)')
    for f in sorted(glob.glob(os.path.join(d1, '*.wav'))):
        if os.path.exists(os.path.join(da, os.path.basename(f) + '.npy')): continue
        x, _ = sf.read(f, dtype='float32')
        f0 = rm.infer_from_audio(x, thred=0.03) if rm else f0_autocorr(x)
        b = os.path.basename(f)
        np.save(os.path.join(db, b + '.npy'), f0.astype(np.float32), allow_pickle=False)
        np.save(os.path.join(da, b + '.npy'), coarse(f0.astype(np.float64)), allow_pickle=False)


def load_cv():
    import torch
    from transformers import HubertModel, HubertConfig
    if os.environ.get('CV_RANDOM'):
        m = HubertModel(HubertConfig())   # только для проверки кода без интернета
    else:
        try:
            m = HubertModel.from_pretrained(CV_NAME)
        except Exception as e:
            print('ContentVec не скачался (', e, ') — беру facebook/hubert-base-ls960')
            m = HubertModel.from_pretrained('facebook/hubert-base-ls960')
    return m.eval()


def stage_feat():
    import torch
    d1, d3 = os.path.join(LOG, '1_16k_wavs'), os.path.join(LOG, '3_feature768')
    os.makedirs(d3, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = load_cv().to(dev)
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(d1, '*.wav'))):
            if os.path.exists(os.path.join(d3, os.path.basename(f)[:-4] + '.npy')): continue
            x, _ = sf.read(f, dtype='float32')
            y = m(torch.from_numpy(x)[None].to(dev)).last_hidden_state[0].float().cpu().numpy()
            np.save(os.path.join(d3, os.path.basename(f)[:-4] + '.npy'), y, allow_pickle=False)
    print('feat: готово', len(os.listdir(d3)))


def final_weights():
    """готовые веса: из assets/weights, иначе — из последней контрольной точки G (если обучение оборвалось на последней эпохе)"""
    import torch
    fin = os.path.join(CK, 'renri_final.pth'), os.path.join('assets', 'weights', EXP + '.pth')
    t0 = float(os.environ.get('RUN_T0', '0'))
    for f in fin:
        if os.path.exists(f) and (f == fin[0] or os.path.getmtime(f) >= t0): return f
    g = sorted(glob.glob(os.path.join(LOG, 'G_*.pth')) + glob.glob(os.path.join(CK, 'G_*.pth')), key=os.path.getmtime)
    if not g: return None
    from infer.lib.train.utils import HParams
    from infer.lib.train.process_ckpt import savee
    ck = torch.load(g[-1], map_location='cpu', weights_only=False)
    cfg = json.load(open(os.path.join(LOG, 'config.json'))); hps = HParams(**cfg)
    os.makedirs(os.path.join('assets', 'weights'), exist_ok=True)
    print(savee(ck['model'], '40k', 1, EXP, ck.get('iteration', 0), 'v2', hps))
    return fin[1]


def stage_train(epochs, batch):
    if os.path.exists(os.path.join(CK, 'renri_final.pth')):
        print('train: уже обучено (веса на Диске) — пропускаю'); return
    os.makedirs(CK, exist_ok=True)
    for f in glob.glob(os.path.join(CK, '[GD]_*.pth')):   # продолжить с места после отключения
        if CK != LOG: shutil.copy(f, LOG); print('продолжаю с', os.path.basename(f))
    gt, fe, fa, fb = [os.path.join(LOG, x) for x in ['0_gt_wavs', '3_feature768', '2a_f0', '2b-f0nsf']]
    names = set(x[:-4] for x in os.listdir(gt)) & set(x[:-4] for x in os.listdir(fe)) & set(x[:-8] for x in os.listdir(fa)) & set(x[:-8] for x in os.listdir(fb))
    lines = ['%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|0' % (gt, n, fe, n, fa, n, fb, n) for n in sorted(names)]
    mu = os.path.join(ROOT, 'logs', 'mute')
    for _ in range(2):
        lines.append('%s/0_gt_wavs/mute40k.wav|%s/3_feature768/mute.npy|%s/2a_f0/mute.wav.npy|%s/2b-f0nsf/mute.wav.npy|0' % (mu, mu, mu, mu))
    import random; random.shuffle(lines)
    open(os.path.join(LOG, 'filelist.txt'), 'w').write('\n'.join(lines))
    cfg = json.load(open(os.path.join(ROOT, 'configs', 'v1', '40k.json')))
    import torch
    if not torch.cuda.is_available():
        cfg['train']['fp16_run'] = False
    cfg['train']['log_interval'] = int(os.environ.get('LOG_INTERVAL', cfg['train']['log_interval']))
    json.dump(cfg, open(os.path.join(LOG, 'config.json'), 'w'), indent=2)
    pg, pd = os.path.join('assets', 'pretrained_v2', 'f0G40k.pth'), os.path.join('assets', 'pretrained_v2', 'f0D40k.pth')
    pre = (['-pg', pg, '-pd', pd] if os.path.exists(pg) and os.path.exists(pd) else [])
    print('train: кусков', len(names), 'предобученная база' if pre else 'БЕЗ базы (только проверка)')
    cmd = [sys.executable, 'infer/modules/train/train.py', '-e', EXP, '-sr', '40k', '-f0', '1', '-bs', str(batch), '-g', '0' if torch.cuda.is_available() else '',
           '-te', str(epochs), '-se', os.environ.get('SAVE_EVERY', '15'), *pre, '-l', '1', '-c', '0', '-sw', '0', '-v', 'v2']
    if not torch.cuda.is_available():
        i = cmd.index('-g'); del cmd[i:i + 2]
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'); os.environ['RUN_T0'] = str(time.time() if 'time' in globals() else __import__('time').time())
    import threading, time
    stop = [False]
    def sync():   # каждую минуту — свежие контрольные точки на Диск
        seen = {}
        while not stop[0]:
            time.sleep(60)
            for f in glob.glob(os.path.join(LOG, '[GD]_*.pth')):
                m = os.path.getmtime(f)
                if CK != LOG and seen.get(f) != m and time.time() - m > 20:
                    try: shutil.copy(f, CK + '/'); seen[f] = m; print('💾 сохранено на Диск:', os.path.basename(f), flush=True)
                    except Exception as e: print('Диск:', e)
    th = threading.Thread(target=sync, daemon=True); th.start()
    lim = float(os.environ.get('TRAIN_SECONDS', '0')) or None   # лимит времени (GitHub: задача ≤6 ч) — по нему берём последнюю контрольную точку
    import signal as sg
    p = subprocess.Popen(cmd, env=env, start_new_session=True)
    try: p.wait(timeout=lim)
    except subprocess.TimeoutExpired:
        print('⏱ время обучения вышло — беру последнюю контрольную точку', flush=True)
        try: os.killpg(p.pid, sg.SIGTERM); time.sleep(10); os.killpg(p.pid, sg.SIGKILL)
        except Exception: pass
    finally: stop[0] = True
    w = final_weights()
    if not w: raise SystemExit('нет весов после обучения (обучение упало — смотри ошибку выше)')
    shutil.copy(w, os.path.join(CK, 'renri_final.pth')); print('✅ обучение готово, веса сохранены на Диск')


def patch_attn():
    """RVC 2.2: в относительном внимании длина замораживается int()/max() при трассировке → ONNX работает только на длине экспорта.
    Заменяем на версии, где длина остаётся символьной."""
    import torch.nn.functional as F
    from infer.lib.infer_pack import attentions as A
    def get_rel(self, rel, length):
        w = self.window_size
        pad = F.pad(rel, [0, 0, length, length, 0, 0])
        return pad[:, w + 1: w + 2 * length]
    def rel2abs(self, x):
        b, h, l, _ = x.size()
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        xf = x.reshape([b, h, l * 2 * l])
        xf = F.pad(xf, [0, l - 1, 0, 0, 0, 0])
        return xf.reshape([b, h, l + 1, 2 * l - 1])[:, :, :l, l - 1:]
    def abs2rel(self, x):
        b, h, l, _ = x.size()
        x = F.pad(x, [0, l - 1, 0, 0, 0, 0, 0, 0])
        xf = x.reshape([b, h, l * l + l * (l - 1)])
        xf = F.pad(xf, [l, 0, 0, 0, 0, 0])
        return xf.reshape([b, h, l, 2 * l])[:, :, :, 1:]
    A.MultiHeadAttention._get_relative_embeddings = get_rel
    A.MultiHeadAttention._relative_position_to_absolute_position = rel2abs
    A.MultiHeadAttention._absolute_position_to_relative_position = abs2rel


def stage_export():
    import torch
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    patch_attn()
    from infer.modules.onnx.export import export_onnx
    w = final_weights()
    os.makedirs('out', exist_ok=True)
    export_onnx(w, 'out/rvc_f32.onnx')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from onnx_half import half_store
    half_store('out/rvc_f32.onnx', 'out/rvc.onnx'); os.remove('out/rvc_f32.onnx')   # веса в float16 (их и учили в half) — файл ~56 МБ
    print('rvc.onnx', os.path.getsize('out/rvc.onnx') // 1e6, 'МБ')
    # ContentVec → ONNX (вход: звук 16 кГц [1,N]; выход: признаки 50 кадров/с [1,T,768]), затем int8 для телефона
    m = load_cv()
    class W(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return s.m(x).last_hidden_state
    torch.onnx.export(W(m), (torch.randn(1, 32000),), 'out/contentvec_f32.onnx', input_names=['audio'], output_names=['feats'],
                      dynamic_axes={'audio': [1], 'feats': [1]}, opset_version=17, do_constant_folding=True)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic('out/contentvec_f32.onnx', 'out/contentvec.onnx', weight_type=QuantType.QInt8)
    os.remove('out/contentvec_f32.onnx')
    print('contentvec.onnx', os.path.getsize('out/contentvec.onnx') // 1e6, 'МБ')


def stage_check(wav16k_path=None):
    """сквозная проверка ONNX: кусок из датасета → признаки → RVC с той же высотой → звук"""
    import onnxruntime as ort
    cv = ort.InferenceSession('out/contentvec.onnx'); rv = ort.InferenceSession('out/rvc.onnx')
    f = wav16k_path or sorted(glob.glob(os.path.join(LOG, '1_16k_wavs', '*.wav')))[0]
    x, _ = sf.read(f, dtype='float32')
    fe = cv.run(None, {'audio': x[None]})[0][0]
    fe = np.repeat(fe, 2, axis=0)                    # 50 → 100 кадров/с
    f0 = f0_autocorr(x)[:len(fe)]; T = min(len(fe), len(f0)); fe, f0 = fe[:T], f0[:T]
    y = rv.run(None, {'phone': fe[None].astype(np.float32), 'phone_lengths': np.array([T], np.int64), 'pitch': coarse(f0.astype(np.float64))[None],
                      'pitchf': f0[None].astype(np.float32), 'ds': np.array([0], np.int64), 'rnd': np.random.randn(1, 192, T).astype(np.float32) * 0.66666})[0].reshape(-1)
    sf.write('out/check.wav', y, SR); print('check: вход', len(x) / 16000, 'с → выход', len(y) / SR, 'с')


def stage_demo(piper_dir, ids_json):
    """русская строка: Piper (женский русский) → признаки → RVC Renri с высотой ×1.5 — слушать в блокноте"""
    import onnxruntime as ort
    pv = ort.InferenceSession(glob.glob(os.path.join(piper_dir, '*.onnx'))[0]); cv = ort.InferenceSession('out/contentvec.onnx'); rv = ort.InferenceSession('out/rvc.onnx')
    outs = []
    for ids in json.load(open(ids_json)):
        ids = np.array([ids], np.int64)
        a = pv.run(None, {'input': ids, 'input_lengths': np.array([ids.shape[1]], np.int64), 'scales': np.array([0.667, 1.0, 0.8], np.float32)})[0].reshape(-1)
        x = resample(a, 22050, 16000)
        fe = np.repeat(cv.run(None, {'audio': x[None]})[0][0], 2, axis=0)
        f0 = f0_autocorr(x) * 1.5; T = min(len(fe), len(f0)); fe, f0 = fe[:T], f0[:T]
        y = rv.run(None, {'phone': fe[None].astype(np.float32), 'phone_lengths': np.array([T], np.int64), 'pitch': coarse(f0.astype(np.float64))[None],
                          'pitchf': f0[None].astype(np.float32), 'ds': np.array([0], np.int64), 'rnd': np.random.randn(1, 192, T).astype(np.float32) * 0.66666})[0].reshape(-1)
        outs += [y, np.zeros(SR // 3, np.float32)]
    sf.write('out/demo_ru.wav', np.concatenate(outs), SR); print('demo: out/demo_ru.wav')


def piper_force_fn():
    import onnx
    from onnx import helper, TensorProto
    def force(src, dst):
        m = onnx.load(src); g = m.graph
        ceil = [n for n in g.node if n.op_type == 'Ceil']; assert len(ceil) == 1, 'Ceil не один'
        c = ceil[0].output[0]
        g.input.append(helper.make_tensor_value_info('durations', TensorProto.FLOAT, [1, 1, 'phonemes']))
        zero = helper.make_tensor('dz_zero', TensorProto.FLOAT, [], [0.0]); g.initializer.append(zero)
        used = 'dur_used'
        nodes = [helper.make_node('Less', ['durations', 'dz_zero'], ['dur_isneg'], name='dur_less'),
                 helper.make_node('Where', ['dur_isneg', c, 'durations'], [used], name='dur_where')]
        for n in g.node:
            if n is ceil[0]: continue
            for i, x in enumerate(n.input):
                if x == c: n.input[i] = used
        idx = list(g.node).index(ceil[0])
        for k, n in enumerate(nodes): g.node.insert(idx + 1 + k, n)
        g.output.append(helper.make_tensor_value_info(c, TensorProto.FLOAT, [1, 1, 'phonemes']))
        g.output[-1].name = c
        onnx.checker.check_model(m); onnx.save(m, dst)
    return force


def stage_pack(out_zip, piper_dir=None):
    meta = {'kind': 'renri-ru', 'ver': 1, 'rvc_sr': SR, 'cv_rate': 50, 'piper_sr': 22050, 'exp': EXP}
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_STORED) as z:
        z.writestr('meta.json', json.dumps(meta))
        z.write('out/rvc.onnx', 'rvc.onnx'); z.write('out/contentvec.onnx', 'contentvec.onnx')
        if piper_dir:
            src = [f for f in glob.glob(os.path.join(piper_dir, '*.onnx')) if 'force' not in f][0]
            piper_force_fn()(src, 'out/ru_force.onnx')     # Piper с управляемыми длительностями (вход durations)
            z.write('out/ru_force.onnx', 'piper/ru.onnx')
            z.write(glob.glob(os.path.join(piper_dir, '*.onnx.json'))[0], 'piper/ru.onnx.json')
    print('пак', out_zip, os.path.getsize(out_zip) // 1e6, 'МБ')


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'prep': stage_prep(sys.argv[2])
    elif st == 'f0': stage_f0()
    elif st == 'feat': stage_feat()
    elif st == 'train': stage_train(int(sys.argv[2]), int(sys.argv[3]))
    elif st == 'export': stage_export()
    elif st == 'check': stage_check()
    elif st == 'demo': stage_demo(sys.argv[2], sys.argv[3])
    elif st == 'pack': stage_pack(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
