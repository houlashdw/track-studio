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
    void installPitch() { if (pstate.equals("installed") || pstate.equals("downloading") || !state.equals("installed") && !state.equals("ready")) return; new Thread(this::installPitchSync).start(); }

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
    float[] pitchCurve(String[] names, long[] dur, float[] f0, double expr, long steps) throws Exception {
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
                    for (int i = 0; i < T && b.hasRemaining(); i++) { float q = b.get(); o[i] = Float.isFinite(q) ? (float) (440.0 * Math.pow(2, (q - 69) / 12.0)) : f0[i]; }
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

    float[] sing(JSONObject q) throws Exception {
        try { load(); } catch (Throwable t) { android.util.Log.e("TrackStudio", "ds load failed", t); ac = null; throw new Exception("модель не загрузилась: " + t); }
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
            try { loadPitch(); if (pp != null) { String[] nm = new String[n]; for (int i = 0; i < n; i++) nm[i] = P.getString(i); f0 = pitchCurve(nm, dur, f0, expr, q.optLong("psteps", 8)); } }
            catch (Throwable t) { android.util.Log.w("TrackStudio", "pitch", t); }
        }
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
}
