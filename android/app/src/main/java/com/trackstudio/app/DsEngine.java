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
import java.util.HashMap;
import java.util.Map;
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
    OrtSession ac, vc;
    final Map<String, Integer> ph = new HashMap<>();
    final Map<String, float[]> emb = new HashMap<>();
    final ExecutorService run = Executors.newSingleThreadExecutor();
    final Map<String, Future<float[]>> jobs = new ConcurrentHashMap<>();
    int nid = 0;

    DsEngine(File base) {
        dir = new File(base, "ds");
        if (new File(dir, "ok").exists()) state = "installed";
    }

    synchronized String status() {
        try {
            JSONObject o = new JSONObject();
            o.put("state", state);
            o.put("progress", progress);
            o.put("loaded", ac != null);
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
                new FileOutputStream(new File(dir, "ok")).close();
                progress = 1;
                state = "installed";
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

    synchronized void load() throws Exception {
        if (ac != null) return;
        if (!new File(dir, "ok").exists()) throw new IllegalStateException("модель не скачана");
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        so.setIntraOpNumThreads(Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
        so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
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
    }

    synchronized String submit(String json) {
        String id = "j" + (++nid);
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
        load();
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
