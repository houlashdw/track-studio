package com.trackstudio.app;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Певческая модель DiffSinger (голос «Canary», tigermeat / JAE VOCAL PROJECT, CC BY-NC-ND 4.0 — только личное некоммерческое использование).
 * Модель качается один раз прямо с релиза автора и работает нативно (onnxruntime-android): быстрее и стабильнее, чем в браузере.
 */
public class DsEngine {
    static final String PACK = "https://github.com/spicytigermeat/canary_diffsinger/releases/download/v106/Canary_DS_v106_PACK.zip";
    static final long PACK_BYTES = 544_000_000L;

    final File dir;
    volatile String state = "none";
    volatile double progress = 0;
    OrtEnvironment env;
    volatile OrtSession ac, vc;
    volatile OrtSession pl, pp;                  /* модель высоты: linguistic + pitch (живые подъезды, вибрато, переходы) */
    final Map<String, Integer> pph = new HashMap<>();
    float[] pemb;
    volatile String pstate = "none";
    final Map<String, Integer> ph = new HashMap<>();
    final Map<String, float[]> emb = new HashMap<>();
    final ExecutorService run = Executors.newSingleThreadExecutor((r) -> { Thread t = new Thread(() -> { android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND); r.run(); }, "ds"); return t; });   /* интерфейс важнее */
    final Map<String, Future<float[]>> jobs = new ConcurrentHashMap<>();
    int nid = 0;

    DsEngine(File base) {
        dir = new File(base, "ds");
        if (new File(dir, "ok").exists()) state = "installed";
        if (new File(dir, "pitch/ok").exists()) pstate = "installed";
        if (new File(new File(base.getAbsoluteFile(), "renri"), "ok").exists()) rnState = "installed";
        if (new File(new File(base.getAbsoluteFile(), "ru"), "ok").exists()) ruState = "installed";
    }


    /* ================= Kokoro (речь, рэп) нативно ================= */
    volatile OrtSession kv;
    volatile String kvKey = null;
    int kvElem = 1; String kvDurOut = null;
    FileOutputStream kvOut;
    File kvDir() { File d = new File(dir.getParentFile(), "kv"); d.mkdirs(); return d; }
    String kvStatus() {
        try {
            if (kvKey == null) { File m = new File(kvDir(), "meta.json"); if (m.exists() && new File(kvDir(), "model.onnx").exists()) { JSONObject j = new JSONObject(readText(m)); kvKey = j.optString("key", null); kvElem = j.optInt("elem", 1); kvDurOut = j.optString("durOut", null); } }
            JSONObject o = new JSONObject(); o.put("key", kvKey == null ? JSONObject.NULL : kvKey); o.put("loaded", kv != null); return o.toString();
        } catch (Exception e) { return "{\"key\":null}"; }
    }
    static String readText(File f) throws IOException { byte[] b = new byte[(int) f.length()]; try (FileInputStream in = new FileInputStream(f)) { int o = 0, k; while (o < b.length && (k = in.read(b, o, b.length - o)) > 0) o += k; } return new String(b, "UTF-8"); }
    synchronized boolean kvBegin(String key) {
        try { if (kvOut != null) kvOut.close(); } catch (IOException e) { }
        try { kvKey = null; new File(kvDir(), "meta.json").delete(); kvOut = new FileOutputStream(new File(kvDir(), "model.part")); return true; } catch (IOException e) { kvOut = null; return false; }
    }
    synchronized boolean kvChunk(String b64) {
        if (kvOut == null) return false;
        try { kvOut.write(android.util.Base64.decode(b64, android.util.Base64.DEFAULT)); return true; } catch (Exception e) { return false; }
    }
    synchronized String kvFinish(String meta) {
        try {
            if (kvOut == null) return "error: нет записи"; kvOut.close(); kvOut = null;
            File part = new File(kvDir(), "model.part"), dst = new File(kvDir(), "model.onnx");
            synchronized (llock) { if (kv != null) { kv.close(); kv = null; } }
            dst.delete(); if (!part.renameTo(dst)) return "error: не записался файл";
            try (FileOutputStream o = new FileOutputStream(new File(kvDir(), "meta.json"))) { o.write(meta.getBytes("UTF-8")); }
            JSONObject j = new JSONObject(meta); kvKey = j.optString("key", null); kvElem = j.optInt("elem", 1); kvDurOut = j.optString("durOut", null);
            return "ok";
        } catch (Exception e) { return "error: " + e.getMessage(); }
    }
    String kvSubmit(String json) {
        String id = "k" + nidA.incrementAndGet();
        jobs.put(id, run.submit(() -> kvGen(new JSONObject(json))));
        return id;
    }
    void dsRelease() {   /* память телефона: две большие модели разом не держим */
        try { if (ac != null) ac.close(); } catch (Exception e) { } try { if (vc != null) vc.close(); } catch (Exception e) { }
        try { if (pl != null) pl.close(); } catch (Exception e) { } try { if (pp != null) pp.close(); } catch (Exception e) { }
        ac = vc = pl = pp = null; if (state.equals("ready")) state = "installed";
    }
    void kvLoad() throws Exception {
        synchronized (llock) {
            if (kv != null) return;
            if (kvKey == null) kvStatus();
            File m = new File(kvDir(), "model.onnx"); if (!m.exists()) throw new IllegalStateException("речевая модель не перенесена");
            dsRelease();
            if (env == null) env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions so = new OrtSession.SessionOptions();
            so.setIntraOpNumThreads(threads()); so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT); so.setMemoryPatternOptimization(false);
            kv = env.createSession(m.getAbsolutePath(), so);
            android.util.Log.i("TrackStudio", "kokoro native ready, elem " + kvElem + ", dur " + kvDurOut);
        }
    }
    OnnxTensor fT(float[] a, long[] shape) throws Exception {
        if (kvElem != 10) return OnnxTensor.createTensor(env, FloatBuffer.wrap(a), shape);
        ByteBuffer bb = ByteBuffer.allocateDirect(a.length * 2).order(ByteOrder.nativeOrder());
        for (float x : a) bb.putShort(android.util.Half.toHalf(x)); bb.rewind();
        return OnnxTensor.createTensor(env, bb, shape, ai.onnxruntime.OnnxJavaType.FLOAT16);
    }
    static float[] toF(OnnxTensor t) {
        ai.onnxruntime.OnnxJavaType ty = t.getInfo().type;
        if (ty == ai.onnxruntime.OnnxJavaType.FLOAT16) { java.nio.ShortBuffer sb = t.getByteBuffer().order(ByteOrder.nativeOrder()).asShortBuffer(); float[] o = new float[sb.remaining()]; for (int i = 0; i < o.length; i++) o[i] = android.util.Half.toFloat(sb.get(i)); return o; }
        if (ty == ai.onnxruntime.OnnxJavaType.INT64) { LongBuffer lb = t.getLongBuffer(); float[] o = new float[lb.remaining()]; for (int i = 0; i < o.length; i++) o[i] = lb.get(i); return o; }
        if (ty == ai.onnxruntime.OnnxJavaType.INT32) { java.nio.IntBuffer ib = t.getIntBuffer(); float[] o = new float[ib.remaining()]; for (int i = 0; i < o.length; i++) o[i] = ib.get(i); return o; }
        FloatBuffer fb = t.getFloatBuffer(); float[] o = new float[fb.remaining()]; fb.get(o); return o;
    }
    /* ответ: [число длительностей, длительности…, звук 24 кГц…] */
    float[] kvGen(JSONObject q) throws Exception {
        try { kvLoad(); } catch (Throwable t) { android.util.Log.e("TrackStudio", "kokoro load", t); throw new Exception("речевая модель не загрузилась: " + t); }
        JSONArray I = q.getJSONArray("ids"), St = q.getJSONArray("style"); int T = I.length();
        long[] ids = new long[T]; for (int i = 0; i < T; i++) ids[i] = I.getLong(i);
        float[] st = new float[256]; for (int i = 0; i < 256; i++) st[i] = (float) St.getDouble(i);
        JSONArray Du = q.optJSONArray("dur"), F = q.optJSONArray("f0");
        long[] dur; long[] dshape; if (Du != null) { dur = new long[T]; for (int i = 0; i < T; i++) dur[i] = Math.max(0, Du.getLong(i)); dshape = new long[]{1, T}; } else { dur = new long[]{1}; dshape = new long[]{1, 1}; }
        float[] f0; if (F != null) { f0 = new float[F.length()]; for (int i = 0; i < f0.length; i++) f0[i] = (float) F.getDouble(i); } else f0 = new float[]{0};
        Map<String, OnnxTensor> in = new HashMap<>();
        try {
            in.put("input_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(ids), new long[]{1, T}));
            in.put("style", OnnxTensor.createTensor(env, FloatBuffer.wrap(st), new long[]{1, 256}));
            in.put("speed", OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{(float) q.optDouble("speed", 1)}), new long[]{1}));
            in.put("dur_in", OnnxTensor.createTensor(env, LongBuffer.wrap(dur), dshape));
            in.put("dur_mix", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{Du != null ? 1 : 0}), new long[]{1}));
            in.put("f0_in", fT(f0, new long[]{1, 1, f0.length}));
            in.put("f0_mix", fT(new float[]{F != null ? (float) q.optDouble("f0mix", 1) : 0f}, new long[]{1}));
            java.util.Set<String> names = kv.getInputNames(); for (String k : new java.util.ArrayList<>(in.keySet())) if (!names.contains(k)) in.remove(k).close();
            try (OrtSession.Result r = kv.run(in)) {
                float[] w = null, d = new float[0];
                for (Map.Entry<String, ai.onnxruntime.OnnxValue> e : r) { String k = e.getKey(); OnnxTensor t = (OnnxTensor) e.getValue();
                    if (k.equals("waveform")) w = toF(t); else if (kvDurOut != null && k.equals(kvDurOut)) d = toF(t); }
                if (w == null) w = toF((OnnxTensor) r.get(0));
                float[] o = new float[1 + d.length + w.length]; o[0] = d.length; System.arraycopy(d, 0, o, 1, d.length); System.arraycopy(w, 0, o, 1 + d.length, w.length);
                for (int i = 1 + d.length; i < o.length; i++) if (!Float.isFinite(o[i])) o[i] = 0f;
                return o;
            }
        } finally { for (OnnxTensor t : in.values()) t.close(); }
    }

    String status() {   /* без блокировки: страница опрашивает статус, пока модель грузится */
        try {
            JSONObject o = new JSONObject();
            o.put("state", state);
            o.put("progress", progress);
            o.put("loaded", ac != null);
            o.put("pitch", pstate);
            o.put("renri", rnState); o.put("rprogress", rnProgress); o.put("ru", ruState); o.put("ruprogress", ruProgress); o.put("sil", silReady() ? "installed" : "none");
            o.put("pprogress", pprogress);
            return o.toString();
        } catch (Exception e) {
            return "{\"state\":\"" + state + "\"}";
        }
    }

    /* скачать пак и вынуть нужные файлы, не сохраняя 0.5 ГБ архива: читаем zip (и zip внутри него) прямо из сети */
    synchronized void install() {
        if (state.equals("downloading") || state.equals("installed") || state.equals("ready")) return;
        state = "downloading";
        progress = 0;
        new Thread(() -> {
            try {
                dir.mkdirs();
                HttpURLConnection c = (HttpURLConnection) new URL(PACK).openConnection();
                c.setInstanceFollowRedirects(true);
                c.setConnectTimeout(30000);
                c.setReadTimeout(60000);
                long total = c.getContentLengthLong() > 0 ? c.getContentLengthLong() : PACK_BYTES;
                final long[] got = {0};
                InputStream raw = new InputStream() {
                    final InputStream in = c.getInputStream();
                    @Override public int read() throws IOException { int b = in.read(); if (b >= 0) tick(1); return b; }
                    @Override public int read(byte[] b, int o, int n) throws IOException { int k = in.read(b, o, n); if (k > 0) tick(k); return k; }
                    void tick(int k) { got[0] += k; progress = Math.min(0.99, got[0] / (double) total); }
                    @Override public void close() throws IOException { in.close(); }
                };
                int found = 0;
                try (ZipInputStream outer = new ZipInputStream(raw)) {
                    ZipEntry e;
                    while ((e = outer.getNextEntry()) != null) {
                        if (e.isDirectory()) continue;
                        String n = e.getName();
                        if (n.toLowerCase().endsWith(".zip")) {
                            ZipInputStream inner = new ZipInputStream(outer);
                            ZipEntry f;
                            while ((f = inner.getNextEntry()) != null) found += take(f.getName(), inner);
                        } else found += take(n, outer);
                    }
                }
                if (!new File(dir, "acoustic.onnx").exists() || !new File(dir, "vocoder.onnx").exists() || !new File(dir, "phonemes.txt").exists())
                    throw new IOException("в паке нет нужных файлов (" + found + ")");
                installPitchSync();   /* модель высоты — сразу, кусками из того же пака */
                new FileOutputStream(new File(dir, "ok")).close();
                progress = 1;
                state = "installed";
                preload();
            } catch (Exception ex) {
                state = "error: " + ex.getMessage();
            }
        }).start();
    }

    int take(String name, InputStream in) throws IOException {
        String n = name.replace('\\', '/');
        String out = null;
        if (n.endsWith("dsacoustic/acoustic.onnx")) out = "acoustic.onnx";
        else if (n.endsWith("dsacoustic/phonemes.txt")) out = "phonemes.txt";
        else if (n.contains("dsacoustic/") && n.endsWith(".emb")) out = n.substring(n.lastIndexOf('/') + 1);
        else if (n.contains("dsvocoder/") && n.endsWith(".onnx")) out = "vocoder.onnx";
        else if (n.endsWith("extra/LICENSE.md")) out = "LICENSE.md";
        if (out == null) return 0;
        File f = new File(dir, out + ".part");
        try (OutputStream o = new FileOutputStream(f)) {
            byte[] buf = new byte[1 << 16];
            int k;
            while ((k = in.read(buf)) > 0) o.write(buf, 0, k);
        }
        File dst = new File(dir, out);
        dst.delete();
        if (!f.renameTo(dst)) throw new IOException("не записался " + out);
        return 1;
    }


    /* ---------- докачка модели высоты: только нужные файлы из пака, по кусочкам (Range), без 0.5 ГБ архива ---------- */
    static byte[] range(long a, long b) throws IOException {
        for (int t = 0; ; t++) {
            try {
                HttpURLConnection c = (HttpURLConnection) new URL(PACK).openConnection();
                c.setInstanceFollowRedirects(true); c.setConnectTimeout(30000); c.setReadTimeout(60000);
                c.setRequestProperty("Range", "bytes=" + a + "-" + b);
                try (InputStream in = c.getInputStream()) {
                    byte[] o = new byte[(int) (b - a + 1)]; int k = 0, n;
                    while (k < o.length && (n = in.read(o, k, o.length - k)) > 0) k += n;
                    if (k != o.length) throw new IOException("обрыв " + k + "/" + o.length);
                    return o;
                }
            } catch (IOException e) { if (t >= 3) throw e; try { Thread.sleep(2000L * (t + 1)); } catch (InterruptedException ie) { throw e; } }
        }
    }
    static int u16(byte[] b, int i) { return (b[i] & 0xff) | ((b[i + 1] & 0xff) << 8); }
    static long u32(byte[] b, int i) { return (u16(b, i) | ((long) u16(b, i + 2) << 16)) & 0xffffffffL; }
    static long[] eocd(byte[] tail) { for (int i = tail.length - 22; i >= 0; i--) if (tail[i] == 0x50 && tail[i + 1] == 0x4b && tail[i + 2] == 5 && tail[i + 3] == 6) return new long[]{u32(tail, i + 12), u32(tail, i + 16)}; return null; }
    /* записи центрального каталога zip, лежащего в [base, base+len) */
    static List<Object[]> entries(long base, long len) throws IOException {
        byte[] tail = range(base + len - 65536, base + len - 1); long[] e = eocd(tail); if (e == null) throw new IOException("не zip");
        byte[] cd = range(base + e[1], base + e[1] + e[0] - 1); List<Object[]> out = new ArrayList<>(); int p = 0;
        while (p + 46 <= cd.length && u32(cd, p) == 0x02014b50L) {
            int meth = u16(cd, p + 10), nl = u16(cd, p + 28), el = u16(cd, p + 30), cl = u16(cd, p + 32);
            long csz = u32(cd, p + 20), lo = u32(cd, p + 42); String name = new String(cd, p + 46, nl, "UTF-8");
            out.add(new Object[]{name, meth, csz, lo}); p += 46 + nl + el + cl;
        }
        return out;
    }
    static long dataStart(long base, long lo) throws IOException { byte[] h = range(base + lo, base + lo + 29); return base + lo + 30 + u16(h, 26) + u16(h, 28); }
    void fetchEntry(long base, Object[] en, File dst, long[] done, long total) throws IOException {
        long ds = dataStart(base, (Long) en[3]), csz = (Long) en[2]; File part = new File(dst.getPath() + ".part");
        try (OutputStream o = new FileOutputStream(part)) {
            OutputStream sink = o; Inflater inf = null;
            java.io.ByteArrayOutputStream tmp = null;
            for (long s = ds; s < ds + csz; ) {
                long e = Math.min(ds + csz, s + (8L << 20)) - 1; byte[] b = range(s, e);
                if ((Integer) en[1] == 8) {
                    if (inf == null) inf = new Inflater(true);
                    inf.setInput(b); byte[] buf = new byte[1 << 16];
                    try { int n; while ((n = inf.inflate(buf)) > 0) o.write(buf, 0, n); } catch (java.util.zip.DataFormatException x) { throw new IOException(x); }
                } else o.write(b);
                done[0] += b.length; pprogress = Math.min(0.99, done[0] / (double) total); s = e + 1;
            }
            if (inf != null) inf.end();
        }
        dst.delete(); if (!part.renameTo(dst)) throw new IOException("не записался " + dst.getName());
    }
    volatile double pprogress = 0;
    final Object plock = new Object();
    void installPitchSync() {
        synchronized (plock) { if (pstate.equals("installed") || pstate.equals("downloading")) return; pstate = "downloading"; pprogress = 0; }
        try {
            File pd = new File(dir, "pitch"); pd.mkdirs();
            Object[] inner = null;
            for (Object[] en : entries(0, PACK_LEN())) if (((String) en[0]).endsWith("Canary_DS_v106.zip")) inner = en;
            if (inner == null) throw new IOException("нет архива голоса");
            long base = dataStart(0, (Long) inner[3]), len = (Long) inner[2];
            List<Object[]> want = new ArrayList<>(); long total = 0;
            for (Object[] en : entries(base, len)) { String n = (String) en[0]; if (n.startsWith("dspitch/files/") && !n.endsWith("/")) { want.add(en); total += (Long) en[2]; } }
            long[] done = {0};
            for (Object[] en : want) { String n = (String) en[0]; fetchEntry(base, en, new File(pd, n.substring(n.lastIndexOf('/') + 1)), done, total); }
            for (String f : new String[]{"linguistic.onnx", "pitch.onnx", "phonemes.txt", "canary.emb"}) if (!new File(pd, f).exists()) throw new IOException("нет " + f);
            new FileOutputStream(new File(pd, "ok")).close(); pprogress = 1; pstate = "installed";
        } catch (Exception ex) { pstate = "error: " + ex.getMessage(); android.util.Log.w("TrackStudio", "pitch install", ex); }
    }
    static long packLen = -1;
    static synchronized long PACK_LEN() throws IOException {
        if (packLen > 0) return packLen;
        HttpURLConnection c = (HttpURLConnection) new URL(PACK).openConnection(); c.setInstanceFollowRedirects(true); c.setRequestMethod("HEAD"); c.setConnectTimeout(30000);
        packLen = c.getContentLengthLong(); c.disconnect(); if (packLen <= 0) throw new IOException("размер пака неизвестен"); return packLen;
    }
    void installPitch() { if (pstate.equals("installed") || pstate.equals("downloading") || !state.equals("installed") && !state.equals("ready") && !rnState.equals("installed")) return;   /* Renri тоже поёт живее с моделью высоты (~125 МБ кусками), Canary для этого не нужен */ new Thread(this::installPitchSync).start(); }

    void loadPitch() throws Exception { synchronized (llock) { loadPitch0(); } }
    void loadPitch0() throws Exception {
        if (pp != null || !new File(dir, "pitch/ok").exists()) return;
        File pd = new File(dir, "pitch");
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        so.setIntraOpNumThreads(threads()); so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT); so.setMemoryPatternOptimization(false);
        pl = env.createSession(new File(pd, "linguistic.onnx").getAbsolutePath(), so);
        try (BufferedReader r = new BufferedReader(new FileReader(new File(pd, "phonemes.txt")))) { String l; int i = 0; while ((l = r.readLine()) != null) { l = l.trim(); if (l.isEmpty()) continue; pph.put(l, i++); } }
        byte[] b = new byte[(int) new File(pd, "canary.emb").length()];
        try (FileInputStream in = new FileInputStream(new File(pd, "canary.emb"))) { int o = 0, k; while (o < b.length && (k = in.read(b, o, b.length - o)) > 0) o += k; }
        FloatBuffer fb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer(); pemb = new float[fb.remaining()]; fb.get(pemb);
        pp = env.createSession(new File(pd, "pitch.onnx").getAbsolutePath(), so);
        android.util.Log.i("TrackStudio", "ds pitch ready");
    }
    static int threads() { int c = Runtime.getRuntime().availableProcessors(); return Math.max(2, Math.min(4, c - 2)); }   /* пара ядер остаётся телефону и интерфейсу */

    /* живая высота: ноты из нашей мелодии → модель высоты рисует кривую как поёт человек */
    float[] pitchCurve(String[] names, long[] dur, float[] f0, double expr, long steps, double tune) throws Exception {
        int n = names.length, T = f0.length; long[] tok = new long[n]; Integer sp = pph.get("SP");
        for (int i = 0; i < n; i++) { Integer t = pph.get(names[i]); tok[i] = t != null ? t : (sp != null ? sp : 0); }
        float[] m = new float[T]; int[] r = new int[T];
        for (int i = 0; i < T; i++) { m[i] = f0[i] > 0 ? (float) (69 + 12 * Math.log(f0[i] / 440.0) / Math.log(2)) : 60f; r[i] = Math.round(m[i]); }
        List<int[]> runs = new ArrayList<>();
        for (int i = 0; i < T; i++) { if (!runs.isEmpty() && runs.get(runs.size() - 1)[0] == r[i]) runs.get(runs.size() - 1)[1]++; else runs.add(new int[]{r[i], 1}); }
        List<int[]> nt = new ArrayList<>();
        for (int[] q : runs) { if (!nt.isEmpty() && (q[1] < 4 || nt.get(nt.size() - 1)[0] == q[0])) nt.get(nt.size() - 1)[1] += q[1]; else nt.add(new int[]{q[0], q[1]}); }
        if (nt.size() > 1 && nt.get(0)[1] < 4) { nt.get(1)[1] += nt.get(0)[1]; nt.remove(0); }
        int N = nt.size(); float[] nm = new float[N]; boolean[] nr = new boolean[N]; long[] nd = new long[N];
        for (int i = 0; i < N; i++) { nm[i] = nt.get(i)[0]; nd[i] = nt.get(i)[1]; }
        float[] ex = new float[T], spk = new float[T * 256]; boolean[] rt = new boolean[T];
        for (int i = 0; i < T; i++) { ex[i] = (float) expr; rt[i] = true; System.arraycopy(pemb, 0, spk, i * 256, 256); }
        Map<String, OnnxTensor> li = new HashMap<>(), pi = new HashMap<>();
        try {
            li.put("tokens", OnnxTensor.createTensor(env, LongBuffer.wrap(tok), new long[]{1, n}));
            li.put("ph_dur", OnnxTensor.createTensor(env, LongBuffer.wrap(dur), new long[]{1, n}));
            try (OrtSession.Result lr = pl.run(li)) {
                OnnxTensor enc = (OnnxTensor) lr.get(0);
                pi.put("ph_dur", OnnxTensor.createTensor(env, LongBuffer.wrap(dur), new long[]{1, n}));
                pi.put("note_midi", OnnxTensor.createTensor(env, FloatBuffer.wrap(nm), new long[]{1, N}));
                pi.put("note_rest", OnnxTensor.createTensor(env, new boolean[][]{nr}));
                pi.put("note_dur", OnnxTensor.createTensor(env, LongBuffer.wrap(nd), new long[]{1, N}));
                pi.put("pitch", OnnxTensor.createTensor(env, FloatBuffer.wrap(m), new long[]{1, T}));
                pi.put("expr", OnnxTensor.createTensor(env, FloatBuffer.wrap(ex), new long[]{1, T}));
                pi.put("retake", OnnxTensor.createTensor(env, new boolean[][]{rt}));
                pi.put("spk_embed", OnnxTensor.createTensor(env, FloatBuffer.wrap(spk), new long[]{1, T, 256}));
                pi.put("steps", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{steps}), new long[]{}));
                Map<String, OnnxTensor> all = new HashMap<>(pi); all.put("encoder_out", enc);
                try (OrtSession.Result pr = pp.run(all)) {
                    FloatBuffer b = ((OnnxTensor) pr.get(0)).getFloatBuffer(); float[] o = new float[T];
                    /* строй: середины нот подтягиваем к ноте (голос не «плавает»), подъезды в начале ноты оставляем живыми */
                    int[] noteOf = new int[T], into = new int[T]; { int f = 0; for (int k = 0; k < N; k++) for (int j = 0; j < nd[k] && f < T; j++, f++) { noteOf[f] = k; into[f] = j; } }
                    for (int i = 0; i < T && b.hasRemaining(); i++) { float q = b.get(); if (!Float.isFinite(q)) { o[i] = f0[i]; continue; }
                        double tgt = nm[noteOf[i]], w = into[i] < 5 ? 0 : Math.min(1, (into[i] - 5) / 6.0) * tune; double mm = q + (tgt - q) * w;
                        o[i] = (float) (440.0 * Math.pow(2, (mm - 69) / 12.0)); }
                    return o;
                }
            }
        } finally { for (OnnxTensor t : li.values()) t.close(); for (OnnxTensor t : pi.values()) t.close(); }
    }

    final Object llock = new Object();
    void load() throws Exception { synchronized (llock) { load0(); } }
    void load0() throws Exception {
        if (ac != null) return;
        if (kv != null) { try { kv.close(); } catch (Exception e) { } kv = null; }
        if (!new File(dir, "ok").exists()) throw new IllegalStateException("модель не скачана");
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        so.setIntraOpNumThreads(threads());
        so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);   /* меньше пиковой памяти при загрузке */
        so.setMemoryPatternOptimization(false);
        android.util.Log.i("TrackStudio", "ds load, free mem " + (Runtime.getRuntime().maxMemory() >> 20) + " MB");
        ac = env.createSession(new File(dir, "acoustic.onnx").getAbsolutePath(), so);
        vc = env.createSession(new File(dir, "vocoder.onnx").getAbsolutePath(), so);
        try (BufferedReader r = new BufferedReader(new FileReader(new File(dir, "phonemes.txt")))) {
            String l;
            int i = 0;
            while ((l = r.readLine()) != null) { l = l.trim(); if (l.isEmpty()) continue; ph.put(l, i++); }
        }
        File[] es = dir.listFiles();
        if (es != null) for (File f : es) if (f.getName().endsWith(".emb")) {
            byte[] b = new byte[(int) f.length()];
            try (FileInputStream in = new FileInputStream(f)) { int o = 0, k; while (o < b.length && (k = in.read(b, o, b.length - o)) > 0) o += k; }
            FloatBuffer fb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            float[] v = new float[fb.remaining()];
            fb.get(v);
            String key = f.getName().replace(".emb", "");
            key = key.substring(key.lastIndexOf('_') + 1);
            emb.put(key, v);
        }
        state = "ready";
        android.util.Log.i("TrackStudio", "ds ready");
    }

    /* загрузить модель заранее, в фоне — чтобы первая песня не ждала загрузки 400 МБ */
    void preload() {
        if (ac != null || !new File(dir, "ok").exists()) return;
        run.submit(() -> { try { load(); } catch (Throwable t) { android.util.Log.w("TrackStudio", "ds preload", t); ac = null; } return null; });
    }

    final java.util.concurrent.atomic.AtomicInteger nidA = new java.util.concurrent.atomic.AtomicInteger();
    /* без блокировки: раньше страница (и весь интерфейс) замирала, пока модель грузилась */
    String submit(String json) {
        String id = "j" + nidA.incrementAndGet();
        jobs.put(id, run.submit(() -> sing(new JSONObject(json))));
        return id;
    }

    /* результат: float32 PCM 44.1 кГц (ждёт, пока посчитается) */
    float[] result(String id) throws Exception {
        Future<float[]> f = jobs.remove(id);
        if (f == null) throw new IllegalStateException("нет задачи " + id);
        return f.get();
    }


    /* ================= второй голос: Yamine Renri (DiffSinger старого формата) ================= */
    static final String NSF = "https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-44.1k-hop512-128bin-2024.02/nsf_hifigan_44.1k_hop512_128bin_2024.02.oudep";
    volatile String rnState = "none"; volatile double rnProgress = 0;
    volatile OrtSession rac, rvc; final Map<String, Integer> rph = new HashMap<>();
    File rnDir() { File d = new File(dir.getParentFile(), "renri"); d.mkdirs(); return d; }
    File base0() { return dir.getParentFile(); }
    static void copyOut(InputStream in, File dst, long[] got, long total, DsEngine e) throws IOException {
        File part = new File(dst.getPath() + ".part");
        try (OutputStream o = new FileOutputStream(part)) { byte[] buf = new byte[1 << 16]; int k; while ((k = in.read(buf)) > 0) { o.write(buf, 0, k); if (e != null && total > 0) { got[0] += k; } } }
        dst.delete(); if (!part.renameTo(dst)) throw new IOException("не записался " + dst.getName());
    }
    /* голос — из zip, который пользователь скачал сам (выбран в «Файлах»); вокодер — с релиза openvpi */
    void installRenri(android.content.ContentResolver cr, android.net.Uri uri) {
        if (rnState.equals("busy")) return; rnState = "busy"; rnProgress = 0;
        new Thread(() -> {
            try {
                File d = rnDir(); new File(d, "ok").delete(); long total = 315_000_000L;
                try (android.database.Cursor c = cr.query(uri, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.SIZE); if (i >= 0 && !c.isNull(i)) total = c.getLong(i); } } catch (Exception ex) { }
                final long T = total; final long[] got = {0};
                InputStream raw = new InputStream() { final InputStream in = cr.openInputStream(uri);
                    @Override public int read() throws IOException { int b = in.read(); if (b >= 0) { got[0]++; } return b; }
                    @Override public int read(byte[] b, int o, int n) throws IOException { int k = in.read(b, o, n); if (k > 0) { got[0] += k; rnProgress = Math.min(0.8, 0.8 * got[0] / (double) T); } return k; }
                    @Override public void close() throws IOException { in.close(); } };
                boolean ac = false, ph = false;
                try (ZipInputStream z = new ZipInputStream(raw)) { ZipEntry e;
                    while ((e = z.getNextEntry()) != null) { String n = e.getName().replace('\\', '/'); if (e.isDirectory() || n.contains("/dsdur/") || n.contains("/dsvariance/") || n.contains("/dspitch/")) continue;
                        if (n.endsWith(".onnx") && !ac) { copyOut(z, new File(d, "acoustic.onnx"), new long[1], 0, null); ac = true; }
                        else if (n.endsWith("phonemes.txt") && !ph) { copyOut(z, new File(d, "phonemes.txt"), new long[1], 0, null); ph = true; } } }
                if (!ac || !ph) throw new IOException("в архиве нет модели голоса");
                rnProgress = 0.82;
                HttpURLConnection c = (HttpURLConnection) new URL(NSF).openConnection(); c.setInstanceFollowRedirects(true); c.setConnectTimeout(30000); c.setReadTimeout(60000);
                boolean vc = false;
                try (ZipInputStream z = new ZipInputStream(c.getInputStream())) { ZipEntry e; while ((e = z.getNextEntry()) != null) if (e.getName().endsWith(".onnx")) { copyOut(z, new File(d, "vocoder.onnx"), new long[1], 0, null); vc = true; break; } }
                if (!vc) throw new IOException("вокодер не скачался");
                new FileOutputStream(new File(d, "ok")).close(); rnProgress = 1; rnState = "installed"; installPitch();
            } catch (Exception ex) { rnState = "error: " + ex.getMessage(); android.util.Log.w("TrackStudio", "renri", ex); }
        }).start();
    }
    void rnLoad() throws Exception {
        synchronized (llock) {
            if (rac != null) return;
            File d = rnDir(); if (!new File(d, "ok").exists()) throw new IllegalStateException("голос Renri не установлен");
            if (ac != null) { try { ac.close(); } catch (Exception e) { } try { vc.close(); } catch (Exception e) { } ac = null; vc = null; }   /* память: Canary-акустика не нужна */
            if (kv != null) { try { kv.close(); } catch (Exception e) { } kv = null; }
            if (env == null) env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions so = new OrtSession.SessionOptions(); so.setIntraOpNumThreads(threads()); so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT); so.setMemoryPatternOptimization(false);
            rac = env.createSession(new File(d, "acoustic.onnx").getAbsolutePath(), so);
            rvc = env.createSession(new File(d, "vocoder.onnx").getAbsolutePath(), so);
            rph.clear(); try (BufferedReader r = new BufferedReader(new FileReader(new File(d, "phonemes.txt")))) { String l; int i = 0; while ((l = r.readLine()) != null) { l = l.trim(); if (l.isEmpty()) continue; rph.put(l, i++); } }
        }
    }
    float[] singRenri(JSONObject q, float[] f0, long[] dur) throws Exception {
        rnLoad(); JSONArray R = q.getJSONArray("rph"); int n = R.length(); long[] tok = new long[n]; Integer sp = rph.get("SP");
        for (int i = 0; i < n; i++) { Integer t = rph.get(R.getString(i)); tok[i] = t != null ? t : (sp != null ? sp : 0); }
        int T = f0.length; Map<String, OnnxTensor> in = new HashMap<>();
        try {
            in.put("tokens", OnnxTensor.createTensor(env, LongBuffer.wrap(tok), new long[]{1, n}));
            in.put("durations", OnnxTensor.createTensor(env, LongBuffer.wrap(dur), new long[]{1, n}));
            in.put("f0", OnnxTensor.createTensor(env, FloatBuffer.wrap(f0), new long[]{1, T}));
            in.put("speedup", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{q.optLong("speedup", 100)}), new long[]{}));
            float[] mel; long[] ms;
            try (OrtSession.Result r = rac.run(in)) { OnnxTensor m = (OnnxTensor) r.get(0); ms = m.getInfo().getShape(); FloatBuffer b = m.getFloatBuffer(); mel = new float[b.remaining()]; b.get(mel); }
            Map<String, OnnxTensor> vin = new HashMap<>();
            try { vin.put("mel", OnnxTensor.createTensor(env, FloatBuffer.wrap(mel), ms)); vin.put("f0", OnnxTensor.createTensor(env, FloatBuffer.wrap(f0), new long[]{1, T}));
                try (OrtSession.Result r = rvc.run(vin)) { FloatBuffer b = ((OnnxTensor) r.get(0)).getFloatBuffer(); float[] w = new float[b.remaining()]; b.get(w); for (int i = 0; i < w.length; i++) if (!Float.isFinite(w[i])) w[i] = 0; return w; } }
            finally { for (OnnxTensor t : vin.values()) t.close(); }
        } finally { for (OnnxTensor t : in.values()) t.close(); }
    }

    float[] sing(JSONObject q) throws Exception {
        boolean RN = "renri".equals(q.optString("voice")) && rnState.equals("installed");
        if (!RN) try { load(); } catch (Throwable t) { android.util.Log.e("TrackStudio", "ds load failed", t); ac = null; throw new Exception("модель не загрузилась: " + t); }
        if (RN && env == null) env = OrtEnvironment.getEnvironment();
        JSONArray P = q.getJSONArray("ph"), Dd = q.getJSONArray("dur"), F = q.getJSONArray("f0");
        int n = P.length(), T = F.length();
        long[] tok = new long[n], dur = new long[n];
        long sum = 0;
        Integer sp = ph.get("SP");
        for (int i = 0; i < n; i++) {
            Integer t = ph.get(P.getString(i));
            tok[i] = t != null ? t : (sp != null ? sp : 0);
            dur[i] = Math.max(1, Dd.getLong(i));
            sum += dur[i];
        }
        if (sum != T) throw new IllegalArgumentException("кадров " + sum + ", высоты " + T);
        float[] f0 = new float[T], gender = new float[T], vel = new float[T];
        for (int i = 0; i < T; i++) { f0[i] = (float) F.getDouble(i); vel[i] = 1f; }
        double expr = q.optDouble("expr", 0);
        if (expr > 0) {
            try { loadPitch(); if (pp != null) { String[] nm = new String[n]; for (int i = 0; i < n; i++) nm[i] = P.getString(i); f0 = pitchCurve(nm, dur, f0, expr, q.optLong("psteps", 8), q.optDouble("tune", 0.6)); } }
            catch (Throwable t) { android.util.Log.w("TrackStudio", "pitch", t); }
        }
        if (RN) return singRenri(q, f0, dur);
        if (q.has("gender")) { double g = q.getDouble("gender"); for (int i = 0; i < T; i++) gender[i] = (float) g; }
        /* голос: смесь эмбеддингов (arc/spark/voltage) */
        float[] e = new float[256];
        JSONObject spk = q.optJSONObject("spk");
        double ws = 0;
        if (spk != null) for (java.util.Iterator<String> it = spk.keys(); it.hasNext(); ) { String k = it.next(); float[] v = emb.get(k); if (v == null) continue; double w = spk.getDouble(k); for (int i = 0; i < 256; i++) e[i] += (float) (v[i] * w); ws += w; }
        if (ws == 0) { float[] v = emb.values().iterator().next(); System.arraycopy(v, 0, e, 0, 256); ws = 1; }
        float[] spkT = new float[T * 256];
        for (int t = 0; t < T; t++) for (int i = 0; i < 256; i++) spkT[t * 256 + i] = (float) (e[i] / ws);
        Map<String, OnnxTensor> in = new HashMap<>();
        try {
            in.put("tokens", OnnxTensor.createTensor(env, LongBuffer.wrap(tok), new long[]{1, n}));
            in.put("durations", OnnxTensor.createTensor(env, LongBuffer.wrap(dur), new long[]{1, n}));
            in.put("f0", OnnxTensor.createTensor(env, FloatBuffer.wrap(f0), new long[]{1, T}));
            in.put("gender", OnnxTensor.createTensor(env, FloatBuffer.wrap(gender), new long[]{1, T}));
            in.put("velocity", OnnxTensor.createTensor(env, FloatBuffer.wrap(vel), new long[]{1, T}));
            in.put("spk_embed", OnnxTensor.createTensor(env, FloatBuffer.wrap(spkT), new long[]{1, T, 256}));
            in.put("depth", OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{(float) q.optDouble("depth", 0.6)}), new long[]{}));
            in.put("steps", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{q.optLong("steps", 8)}), new long[]{}));
            float[] mel;
            long[] mshape;
            try (OrtSession.Result r = ac.run(in)) {
                OnnxTensor m = (OnnxTensor) r.get(0);
                mshape = m.getInfo().getShape();
                FloatBuffer b = m.getFloatBuffer();
                mel = new float[b.remaining()];
                b.get(mel);
            }
            Map<String, OnnxTensor> vin = new HashMap<>();
            try {
                vin.put("mel", OnnxTensor.createTensor(env, FloatBuffer.wrap(mel), mshape));
                vin.put("f0", OnnxTensor.createTensor(env, FloatBuffer.wrap(f0), new long[]{1, T}));
                try (OrtSession.Result r = vc.run(vin)) {
                    FloatBuffer b = ((OnnxTensor) r.get(0)).getFloatBuffer();
                    float[] w = new float[b.remaining()];
                    b.get(w);
                    return w;
                }
            } finally {
                for (OnnxTensor t : vin.values()) t.close();
            }
        } finally {
            for (OnnxTensor t : in.values()) t.close();
        }
    }
    /* ================= Renri по-русски: Piper (русский голос, длительности задаём мы) → ContentVec → RVC (тембр Renri, наша высота) =================
       пак — zip из Colab-блокнота: rvc.onnx, contentvec.onnx, piper/ru.onnx(+json), meta.json */
    volatile String ruState = "none"; volatile double ruProgress = 0;
    volatile OrtSession pv, cvs, rvs; final Map<String, int[]> ruIds = new HashMap<>(); boolean pvForced = false;
    File ruDir() { File d = new File(dir.getParentFile(), "ru"); d.mkdirs(); return d; }
    /* ===== Silero v5 (русский): поёт сама — длительность и высота каждого звука задаются приложением ===== */
    volatile OrtSession sac, svoc, sdur;
    boolean silReady() { File d = ruDir(); return new File(d, "sil_acoustic.onnx").exists() && new File(d, "sil_vocoder.onnx").exists() && new File(d, "sil_dur.onnx").exists(); }
    void silRelease() { try { if (sac != null) sac.close(); if (svoc != null) svoc.close(); if (sdur != null) sdur.close(); } catch (Exception e) { } sac = svoc = sdur = null; }
    void silLoad() throws Exception {
        synchronized (llock) {
            if (sac != null) return;
            if (env == null) env = OrtEnvironment.getEnvironment();
            if (kv != null) { try { kv.close(); } catch (Exception e) { } kv = null; }
            ruRelease();
            File d = ruDir(); OrtSession.SessionOptions so = new OrtSession.SessionOptions();
            so.setIntraOpNumThreads(threads()); so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT); so.setMemoryPatternOptimization(false);
            sdur = env.createSession(new File(d, "sil_dur.onnx").getAbsolutePath(), so);
            sac = env.createSession(new File(d, "sil_acoustic.onnx").getAbsolutePath(), so);
            svoc = env.createSession(new File(d, "sil_vocoder.onnx").getAbsolutePath(), so);
        }
    }
    static long[] jl(JSONArray a) throws Exception { long[] o = new long[a.length()]; for (int i = 0; i < o.length; i++) o[i] = a.getLong(i); return o; }
    static float[] jf(JSONArray a) throws Exception { float[] o = new float[a.length()]; for (int i = 0; i < o.length; i++) o[i] = (float) a.getDouble(i); return o; }
    /* job "silNat": естественные длительности звуков (кадры 512 @48 кГц). q: ids, spk */
    String silNatSubmit(String json) { String id = "s" + nidA.incrementAndGet(); jobs.put(id, run.submit(() -> {
        silLoad(); JSONObject q = new JSONObject(json); long[] ids = jl(q.getJSONArray("ids")); int n = ids.length; Map<String, OnnxTensor> in = new HashMap<>();
        try { boolean[][] mask = new boolean[1][n]; long[] z = new long[n];
            in.put("text", OnnxTensor.createTensor(env, LongBuffer.wrap(ids), new long[]{1, n}));
            in.put("speaker_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{q.optLong("spk", 2)}), new long[]{1}));
            in.put("mask", OnnxTensor.createTensor(env, mask));
            in.put("type_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(z), new long[]{1, n}));
            in.put("focus_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(z.clone()), new long[]{1, n}));
            try (OrtSession.Result r = sdur.run(in)) { float[] d = toF((OnnxTensor) r.get(0)); for (int i = 0; i < d.length; i++) d[i] = (float) Math.max(0, Math.exp(d[i]) - 1); return d; }
        } finally { for (OnnxTensor t : in.values()) t.close(); } })); return id; }
    /* job "silSing": q: ids, spk, durs (кадры), pitch (норм. высота на звук). Выход: float32 44.1 кГц */
    String silSingSubmit(String json) { String id = "s" + nidA.incrementAndGet(); jobs.put(id, run.submit(() -> {
        silLoad(); JSONObject q = new JSONObject(json); long[] ids = jl(q.getJSONArray("ids")); int n = ids.length; float[] du = jf(q.getJSONArray("durs")), pi = jf(q.getJSONArray("pitch"));
        float[] one = new float[n]; java.util.Arrays.fill(one, 1f); long[] z = new long[n]; Map<String, OnnxTensor> in = new HashMap<>(); float[] mel; long[] ms;
        try { in.put("sequence", OnnxTensor.createTensor(env, LongBuffer.wrap(ids), new long[]{1, n}));
            in.put("speaker_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{q.optLong("spk", 2)}), new long[]{1}));
            in.put("durs_rate", OnnxTensor.createTensor(env, FloatBuffer.wrap(one), new long[]{1, n}));
            in.put("gt_durs", OnnxTensor.createTensor(env, FloatBuffer.wrap(du), new long[]{1, n}));
            in.put("gt_pitch", OnnxTensor.createTensor(env, FloatBuffer.wrap(pi), new long[]{1, 1, n}));
            in.put("pitch_coefs", OnnxTensor.createTensor(env, FloatBuffer.wrap(one.clone()), new long[]{1, n}));
            in.put("type_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(z), new long[]{1, n}));
            in.put("focus_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(z.clone()), new long[]{1, n}));
            try (OrtSession.Result r = sac.run(in)) { OnnxTensor t = (OnnxTensor) r.get(0); mel = toF(t); ms = t.getInfo().getShape(); }
        } finally { for (OnnxTensor t : in.values()) t.close(); in.clear(); }
        float[] a;
        try { in.put("mel", OnnxTensor.createTensor(env, FloatBuffer.wrap(mel), ms));
            try (OrtSession.Result r = svoc.run(in)) { a = toF((OnnxTensor) r.get(0)); } }
        finally { for (OnnxTensor t : in.values()) t.close(); }
        return lin(a, 48000, 44100); })); return id; }
    /* файлы пака: zip целиком ИЛИ отдельные файлы (rvc.onnx, contentvec.onnx, ru.onnx, ru.onnx.json — как их скачал браузер, имена могут быть с «(1)»).
       Можно выбрать все разом или по одному: «готово», когда на месте все четыре */
    static String ruTarget(String name) {
        String n = name.toLowerCase();
        if (n.endsWith(".json")) return "ru.onnx.json";
        if (!n.endsWith(".onnx")) return null;
        if (n.contains("acoustic")) return "sil_acoustic.onnx";   /* Silero v5 (русский певческий путь): acoustic_v5_6_ru, vocoder_v5_6_ru, dur_predictor_v5_6_ru */
        if (n.contains("vocoder")) return "sil_vocoder.onnx";
        if (n.contains("dur_pred")) return "sil_dur.onnx";
        if (n.contains("contentvec")) return "contentvec.onnx";
        if (n.contains("rvc")) return "rvc.onnx";
        if (n.startsWith("ru")) return "ru.onnx";
        return null;
    }
    void installRu(android.content.ContentResolver cr, List<android.net.Uri> uris) {
        if (ruState.equals("busy")) return; ruState = "busy"; ruProgress = 0;
        new Thread(() -> {
            try {
                File d = ruDir(); new File(d, "ok").delete(); long total = 0; List<String> names = new ArrayList<>();
                for (android.net.Uri u : uris) { String nm = "file"; long sz = 60_000_000L;
                    try (android.database.Cursor c = cr.query(u, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.SIZE); if (i >= 0 && !c.isNull(i)) sz = c.getLong(i); int k = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME); if (k >= 0 && !c.isNull(k)) nm = c.getString(k); } } catch (Exception e) { }
                    names.add(nm); total += sz; }
                final long T = Math.max(1, total); final long[] got = {0};
                for (int q = 0; q < uris.size(); q++) {
                    final android.net.Uri u = uris.get(q); String nm = names.get(q);
                    InputStream raw = new InputStream() { final InputStream in = cr.openInputStream(u);
                        @Override public int read() throws IOException { int b = in.read(); if (b >= 0) got[0]++; return b; }
                        @Override public int read(byte[] b, int o, int n) throws IOException { int k = in.read(b, o, n); if (k > 0) { got[0] += k; ruProgress = Math.min(0.99, got[0] / (double) T); } return k; }
                        @Override public void close() throws IOException { in.close(); } };
                    if (nm.toLowerCase().endsWith(".zip")) {
                        try (ZipInputStream z = new ZipInputStream(raw)) { ZipEntry e;
                            while ((e = z.getNextEntry()) != null) { String n = e.getName().replace('\\', '/'); if (e.isDirectory()) continue; String b = n.substring(n.lastIndexOf('/') + 1); String t = ruTarget(b);
                                if (t != null) copyOut(z, new File(d, t), new long[1], 0, null); } }
                    } else { String t = ruTarget(nm); if (t == null) { raw.close(); throw new IOException("непонятный файл: " + nm); } try (InputStream in = raw) { copyOut(in, new File(d, t), new long[1], 0, null); } }
                }
                List<String> miss = new ArrayList<>();
                boolean sil = silReady(); List<String> missS = new ArrayList<>();
                for (String f : new String[]{"sil_acoustic.onnx", "sil_vocoder.onnx", "sil_dur.onnx"}) if (!new File(d, f).exists()) missS.add(f);
                if (!sil) for (String f : new String[]{"ru.onnx", "ru.onnx.json"}) if (!new File(d, f).exists()) miss.add(f);   /* хватает Silero (3 файла) ИЛИ Piper (2 файла) */
                if (!sil && missS.size() < 3) { miss.clear(); for (String f : missS) miss.add(f.replace("sil_acoustic", "acoustic_v5_6_ru").replace("sil_vocoder", "vocoder_v5_6_ru").replace("sil_dur", "dur_predictor_v5_6_ru")); }
                synchronized (llock) { silRelease(); }
                synchronized (llock) { ruRelease(); }
                if (!miss.isEmpty()) { ruState = "error: не хватает " + String.join(", ", miss) + " — выбери и их"; return; }
                new FileOutputStream(new File(d, "ok")).close(); ruProgress = 1; ruState = "installed";
            } catch (Exception ex) { ruState = "error: " + ex.getMessage(); android.util.Log.w("TrackStudio", "ru", ex); }
        }).start();
    }
    void ruRelease() { try { if (pv != null) pv.close(); if (cvs != null) cvs.close(); if (rvs != null) rvs.close(); } catch (Exception e) { } pv = cvs = rvs = null; }
    void ruLoad() throws Exception {
        synchronized (llock) {
            if (pv != null) return;
            if (env == null) env = OrtEnvironment.getEnvironment();
            if (kv != null) { try { kv.close(); } catch (Exception e) { } kv = null; }
            silRelease();
            File d = ruDir(); OrtSession.SessionOptions so = new OrtSession.SessionOptions();
            so.setIntraOpNumThreads(threads()); so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT); so.setMemoryPatternOptimization(false);
            pv = env.createSession(new File(d, "ru.onnx").getAbsolutePath(), so);
            pvForced = pv.getInputNames().contains("durations");
            if (new File(d, "contentvec.onnx").exists() && new File(d, "rvc.onnx").exists()) { cvs = env.createSession(new File(d, "contentvec.onnx").getAbsolutePath(), so); rvs = env.createSession(new File(d, "rvc.onnx").getAbsolutePath(), so); }
            byte[] b = java.nio.file.Files.readAllBytes(new File(d, "ru.onnx.json").toPath());
            JSONObject m = new JSONObject(new String(b, "UTF-8")).getJSONObject("phoneme_id_map"); ruIds.clear();
            for (java.util.Iterator<String> it = m.keys(); it.hasNext(); ) { String k = it.next(); JSONArray a = m.getJSONArray(k); int[] v = new int[a.length()]; for (int i = 0; i < v.length; i++) v[i] = a.getInt(i); ruIds.put(k, v); }
            android.util.Log.i("TrackStudio", "ru ready, forced=" + pvForced);
        }
    }
    /* фонемы → ids как у Piper: ^ _ (ф _)* $ ; tokPh — номер фонемы для каждого id (-1: служебные/паузы) */
    long[] ruTokens(JSONArray ph, List<Integer> tokPh) throws Exception {
        List<Long> ids = new ArrayList<>(); ids.add((long) ruIds.get("^")[0]); tokPh.add(-1); ids.add(0L); tokPh.add(-1);
        for (int i = 0; i < ph.length(); i++) { int[] v = ruIds.get(ph.getString(i)); if (v == null) continue; for (int x : v) { ids.add((long) x); tokPh.add(i); } ids.add(0L); tokPh.add(-1); }
        ids.add((long) ruIds.get("$")[0]); tokPh.add(-1);
        long[] o = new long[ids.size()]; for (int i = 0; i < o.length; i++) o[i] = ids.get(i); return o;
    }
    float[][] piper(long[] ids, float[] dur, float noise, float noiseW) throws Exception {
        int n = ids.length; Map<String, OnnxTensor> in = new HashMap<>();
        try {
            in.put("input", OnnxTensor.createTensor(env, LongBuffer.wrap(ids), new long[]{1, n}));
            in.put("input_lengths", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{n}), new long[]{1}));
            in.put("scales", OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{noise, 1f, noiseW}), new long[]{3}));
            if (pvForced) { float[] d = dur; if (d == null) { d = new float[n]; java.util.Arrays.fill(d, -1f); } in.put("durations", OnnxTensor.createTensor(env, FloatBuffer.wrap(d), new long[]{1, 1, n})); }
            try (OrtSession.Result r = pv.run(in)) {
                float[] a = toF((OnnxTensor) r.get(0)); float[] dp = pvForced ? toF((OnnxTensor) r.get(1)) : new float[0];
                return new float[][]{a, dp};
            }
        } finally { for (OnnxTensor t : in.values()) t.close(); }
    }
    static float[] lin(float[] x, double sr0, double sr1) {
        int n = (int) Math.floor(x.length * sr1 / sr0); float[] o = new float[n]; double k = sr0 / sr1;
        for (int i = 0; i < n; i++) { double p = i * k; int a = (int) p; double w = p - a; o[i] = (float) (x[Math.min(a, x.length - 1)] * (1 - w) + x[Math.min(a + 1, x.length - 1)] * w); }
        return o;
    }
    static float[] f0auto(float[] x, int sr, int hop) {   /* высота исходной речи (автокорреляция), 0 — глухое/тишина */
        int W = 1024, n = x.length / hop + 1; float[] o = new float[n]; double[] s = new double[W];
        for (int f = 0; f < n; f++) { int a = f * hop - W / 2; double mean = 0, e = 0;
            for (int i = 0; i < W; i++) { int j = a + i; s[i] = j >= 0 && j < x.length ? x[j] : 0; mean += s[i]; } mean /= W;
            for (int i = 0; i < W; i++) { s[i] -= mean; e += s[i] * s[i]; }
            if (e < W * 1e-5) continue;
            int lo = sr / 1000, hi = sr / 70; double best = 0; int bk = 0;
            for (int k = lo; k < hi; k++) { double c = 0; for (int i = 0; i + k < W; i++) c += s[i] * s[i + k]; c /= e; if (c > best) { best = c; bk = k; } }
            if (best > 0.5 && bk > 0) o[f] = (float) sr / bk; }
        return o;
    }
    static long coarse(double f) { if (f <= 0) return 1; double mn = 1127 * Math.log(1 + 50 / 700.0), mx = 1127 * Math.log(1 + 1100 / 700.0), m = 1127 * Math.log(1 + f / 700); m = (m - mn) * 254 / (mx - mn) + 1; return Math.max(1, Math.min(255, Math.round(m))); }
    /* job "ruNat": естественные длительности Piper. Выход: [N, tokPh..., dur...] */
    String ruNatSubmit(String json) { String id = "r" + nidA.incrementAndGet(); jobs.put(id, run.submit(() -> {
        ruLoad(); JSONObject q = new JSONObject(json); List<Integer> tp = new ArrayList<>(); long[] ids = ruTokens(q.getJSONArray("ph"), tp);
        float[][] r = piper(ids, null, 0.0f, 0.0f); int n = ids.length; float[] o = new float[1 + 2 * n]; o[0] = n;
        for (int i = 0; i < n; i++) { o[1 + i] = tp.get(i); o[1 + n + i] = i < r[1].length ? r[1][i] : 1; }
        return o; })); return id; }
    /* job "ruSing": строка с нашими длительностями и высотой. q: ph, dur (кадры по 256 @22050 на каждый токен), f0 (Гц по 10 мс; 0 — «как в речи × shift»), shift.
       Выход: float32 44.1 кГц, отсчёт 0 = начало строки */
    String ruSingSubmit(String json) { String id = "r" + nidA.incrementAndGet(); jobs.put(id, run.submit(() -> {
        ruLoad(); JSONObject q = new JSONObject(json); List<Integer> tp = new ArrayList<>(); long[] ids = ruTokens(q.getJSONArray("ph"), tp);
        JSONArray dj = q.optJSONArray("dur"); float[] dur = null; if (dj != null && dj.length() == ids.length) { dur = new float[ids.length]; for (int i = 0; i < dur.length; i++) dur[i] = (float) dj.getDouble(i); }
        float[] a = piper(ids, dur, (float) q.optDouble("noise", 0.45), (float) q.optDouble("noiseW", 0.5))[0];
        if (q.optBoolean("raw", false) || rvs == null || cvs == null) return lin(a, 22050, 44100);   /* речь как есть — высоту и тембр делает приложение */
        float[] x16 = lin(a, 22050, 16000);
        float[] fe; long[] fs;
        Map<String, OnnxTensor> in = new HashMap<>();
        try { in.put("audio", OnnxTensor.createTensor(env, FloatBuffer.wrap(x16), new long[]{1, x16.length}));
            try (OrtSession.Result r = cvs.run(in)) { OnnxTensor t = (OnnxTensor) r.get(0); fe = toF(t); fs = t.getInfo().getShape(); } }
        finally { for (OnnxTensor t : in.values()) t.close(); in.clear(); }
        int T50 = (int) fs[1], C = (int) fs[2], T = Math.min(T50 * 2, x16.length / 160);
        float[] ph = new float[T * C]; for (int t = 0; t < T; t++) System.arraycopy(fe, (t / 2) * C, ph, t * C, C);
        float[] src = f0auto(x16, 16000, 160); JSONArray fj = q.optJSONArray("f0"); double shift = q.optDouble("shift", 1.5);
        float[] pf = new float[T]; long[] pc = new long[T];
        for (int t = 0; t < T; t++) { double s0 = t < src.length ? src[t] : 0, want = fj != null && t < fj.length() ? fj.optDouble(t, 0) : 0;
            double f = s0 > 0 ? (want > 0 ? want : s0 * shift) : 0; pf[t] = (float) f; pc[t] = coarse(f); }
        float[] rnd = new float[192 * T]; java.util.Random R = new java.util.Random(T * 31L); for (int i = 0; i < rnd.length; i++) rnd[i] = (float) (R.nextGaussian() * 0.66666);
        float[] y;
        try { in.put("phone", OnnxTensor.createTensor(env, FloatBuffer.wrap(ph), new long[]{1, T, C}));
            in.put("phone_lengths", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{T}), new long[]{1}));
            in.put("pitch", OnnxTensor.createTensor(env, LongBuffer.wrap(pc), new long[]{1, T}));
            in.put("pitchf", OnnxTensor.createTensor(env, FloatBuffer.wrap(pf), new long[]{1, T}));
            in.put("ds", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{0}), new long[]{1}));
            in.put("rnd", OnnxTensor.createTensor(env, FloatBuffer.wrap(rnd), new long[]{1, 192, T}));
            try (OrtSession.Result r = rvs.run(in)) { y = toF((OnnxTensor) r.get(0)); } }
        finally { for (OnnxTensor t : in.values()) t.close(); }
        return lin(y, 40000, 44100); })); return id; }
}
