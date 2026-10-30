package com.trackstudio.app;

import android.app.Activity;
import android.content.ClipData;
import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.Window;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Трек Студия: студия (assets/www/index.html) в WebView.
 * Страница открывается как https://appassets.androidplatform.net/ — это «безопасный» адрес,
 * поэтому кэш моделей, IndexedDB и загрузки с CDN работают как в Chrome.
 */
public class MainActivity extends Activity {
    static final String HOST = "appassets.androidplatform.net";
    static final String START = "https://" + HOST + "/index.html";
    static final String BG = "#111113";
    static final int PICK = 41;

    WebView web;
    ValueCallback<Uri[]> pending;
    DsEngine ds;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        Window w = getWindow();
        w.setStatusBarColor(Color.parseColor(BG));
        w.setNavigationBarColor(Color.parseColor(BG));
        WebView.setWebContentsDebuggingEnabled(true);

        ds = new DsEngine(getFilesDir());
        new android.os.Handler(android.os.Looper.getMainLooper()).postDelayed(() -> ds.preload(), 4000);
        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor(BG));
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setDatabaseEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(true);
        s.setTextZoom(100);
        s.setUserAgentString(s.getUserAgentString() + " TrackStudioApp/1");
        web.addJavascriptInterface(new Bridge(), "TSAndroid");
        web.setWebViewClient(new Client());
        web.setWebChromeClient(new Chrome());
        setContentView(web);
        web.loadUrl(START);
    }

    /* «Назад» сворачивает приложение, а не закрывает его: трек и настройки остаются на месте */
    @Override
    public void onBackPressed() {
        moveTaskToBack(true);
    }

    @Override
    protected void onDestroy() {
        if (web != null) web.destroy();
        super.onDestroy();
    }

    static String mime(String path) {
        String p = path.toLowerCase();
        if (p.endsWith(".html") || p.endsWith(".htm")) return "text/html";
        if (p.endsWith(".js") || p.endsWith(".mjs")) return "text/javascript";
        if (p.endsWith(".json") || p.endsWith(".webmanifest")) return "application/json";
        if (p.endsWith(".css")) return "text/css";
        if (p.endsWith(".png")) return "image/png";
        if (p.endsWith(".svg")) return "image/svg+xml";
        if (p.endsWith(".wasm")) return "application/wasm";
        if (p.endsWith(".txt") || p.endsWith(".md")) return "text/plain";
        return "application/octet-stream";
    }

    class Client extends WebViewClient {
        @Override
        public WebResourceResponse shouldInterceptRequest(WebView v, WebResourceRequest r) {
            Uri u = r.getUrl();
            if (u == null || !HOST.equals(u.getHost())) return null;
            String p = u.getPath();
            /* певческая модель: готовый звук строки (float32, 44.1 кГц) — запрос ждёт, пока модель досчитает */
            if (p != null && p.startsWith("/ds/out/")) {
                Map<String, String> h = new HashMap<>();
                h.put("Access-Control-Allow-Origin", "*");
                h.put("Cache-Control", "no-store");
                try {
                    float[] w = ds.result(p.substring(8));
                    java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(w.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
                    bb.asFloatBuffer().put(w);
                    return new WebResourceResponse("application/octet-stream", null, 200, "OK", h, new ByteArrayInputStream(bb.array()));
                } catch (Exception e) {
                    String m = String.valueOf(e.getCause() != null ? e.getCause().getMessage() : e.getMessage());
                    return new WebResourceResponse("text/plain", "utf-8", 500, "Error", h, new ByteArrayInputStream(m.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
                }
            }
            if (p == null || p.isEmpty()) p = "/";
            if (p.endsWith("/")) p = p + "index.html";
            try {
                InputStream in = getAssets().open("www" + p);
                String m = mime(p);
                boolean text = m.startsWith("text/") || m.equals("application/json");
                WebResourceResponse res = new WebResourceResponse(m, text ? "utf-8" : null, in);
                Map<String, String> h = new HashMap<>();
                h.put("Cache-Control", "no-cache");
                h.put("Access-Control-Allow-Origin", "*");
                res.setResponseHeaders(h);
                return res;
            } catch (IOException e) {
                return new WebResourceResponse("text/plain", "utf-8", 404, "Not Found", new HashMap<String, String>(), new ByteArrayInputStream(new byte[0]));
            }
        }

        /* ссылки на другие сайты — во внешний браузер */
        @Override
        public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest r) {
            Uri u = r.getUrl();
            if (u != null && HOST.equals(u.getHost())) return false;
            try {
                startActivity(new Intent(Intent.ACTION_VIEW, u));
            } catch (Exception e) {
                // нет приложения для ссылки — просто игнорируем
            }
            return true;
        }
    }

    class Chrome extends WebChromeClient {
        @Override
        public boolean onShowFileChooser(WebView v, ValueCallback<Uri[]> cb, FileChooserParams fp) {
            if (pending != null) pending.onReceiveValue(null);
            pending = cb;
            Intent i = new Intent(Intent.ACTION_GET_CONTENT);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            i.setType("*/*");
            ArrayList<String> types = new ArrayList<>();
            boolean ext = false;
            String[] acc = fp.getAcceptTypes();
            if (acc != null) {
                for (String a : acc) {
                    if (a == null) continue;
                    for (String x : a.split(",")) {
                        x = x.trim();
                        if (x.startsWith(".")) ext = true;
                        else if (x.contains("/")) types.add(x);
                    }
                }
            }
            if (!types.isEmpty() && !ext) i.putExtra(Intent.EXTRA_MIME_TYPES, types.toArray(new String[0]));
            if (fp.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE) i.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
            try {
                startActivityForResult(Intent.createChooser(i, "Выбери файл"), PICK);
            } catch (Exception e) {
                pending = null;
                cb.onReceiveValue(null);
            }
            return true;
        }

        @Override
        public void onPermissionRequest(PermissionRequest r) {
            r.deny();
        }
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        if (req != PICK) {
            super.onActivityResult(req, res, data);
            return;
        }
        ValueCallback<Uri[]> cb = pending;
        pending = null;
        if (cb == null) return;
        Uri[] out = null;
        if (res == RESULT_OK && data != null) {
            ClipData cd = data.getClipData();
            if (cd != null && cd.getItemCount() > 0) {
                out = new Uri[cd.getItemCount()];
                for (int k = 0; k < out.length; k++) out[k] = cd.getItemAt(k).getUri();
            } else if (data.getData() != null) {
                out = new Uri[]{data.getData()};
            }
        }
        cb.onReceiveValue(out);
    }

    static String safeName(String n) {
        String s = n == null ? "" : n.replaceAll("[\\\\/:*?\"<>|\\p{Cntrl}]", "_").trim();
        if (s.isEmpty()) s = "file";
        if (s.length() > 120) s = s.substring(s.length() - 120);
        return s;
    }

    static void copy(InputStream in, OutputStream o) throws IOException {
        byte[] buf = new byte[65536];
        int n;
        while ((n = in.read(buf)) > 0) o.write(buf, 0, n);
    }

    /* сохранить в «Загрузки/TrackStudio»; вернуть content://-адрес (Android 10+) для «Поделиться» */
    Uri saveToDownloads(File f, String name, String mime) throws IOException {
        if (Build.VERSION.SDK_INT >= 29) {
            ContentValues cv = new ContentValues();
            cv.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
            cv.put(MediaStore.MediaColumns.MIME_TYPE, mime);
            cv.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/TrackStudio");
            Uri u = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv);
            if (u == null) throw new IOException("не удалось создать файл в Загрузках");
            try (OutputStream o = getContentResolver().openOutputStream(u); InputStream in = new FileInputStream(f)) {
                if (o == null) throw new IOException("нет доступа к Загрузкам");
                copy(in, o);
            }
            return u;
        }
        File dir = new File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "TrackStudio");
        if (!dir.exists() && !dir.mkdirs()) throw new IOException("не удалось создать папку");
        try (InputStream in = new FileInputStream(f); OutputStream o = new FileOutputStream(new File(dir, name))) {
            copy(in, o);
        }
        return null;
    }

    /** Мост для страницы: файл передаётся кусками base64 (begin → chunk… → finish). */
    class Bridge {
        final Map<String, File> parts = new HashMap<>();
        final Map<String, String[]> meta = new HashMap<>();
        int n = 0;

        @JavascriptInterface
        public synchronized String begin(String name, String mime) {
            String id = "f" + (++n);
            File f = new File(getCacheDir(), "out-" + id);
            if (f.exists()) f.delete();
            parts.put(id, f);
            meta.put(id, new String[]{safeName(name), mime == null || mime.isEmpty() ? "application/octet-stream" : mime});
            return id;
        }

        @JavascriptInterface
        public synchronized boolean chunk(String id, String b64) {
            File f = parts.get(id);
            if (f == null) return false;
            try (FileOutputStream o = new FileOutputStream(f, true)) {
                o.write(Base64.decode(b64, Base64.DEFAULT));
                return true;
            } catch (Exception e) {
                return false;
            }
        }

        @JavascriptInterface
        public synchronized String finish(String id, boolean share) {
            File f = parts.remove(id);
            String[] m = meta.remove(id);
            if (f == null || m == null) return "error: нет файла";
            try {
                Uri uri = saveToDownloads(f, m[0], m[1]);
                if (share && uri != null) {
                    Intent s = new Intent(Intent.ACTION_SEND);
                    s.setType(m[1]);
                    s.putExtra(Intent.EXTRA_STREAM, uri);
                    s.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                    final Intent ch = Intent.createChooser(s, m[0]);
                    runOnUiThread(() -> {
                        try {
                            startActivity(ch);
                        } catch (Exception e) {
                            // меню «Поделиться» недоступно — файл всё равно сохранён
                        }
                    });
                }
                return Build.VERSION.SDK_INT >= 29 ? "Загрузки/TrackStudio/" + m[0] : "Android/data/" + getPackageName() + "/files/Download/TrackStudio/" + m[0];
            } catch (Exception e) {
                return "error: " + e.getMessage();
            } finally {
                f.delete();
            }
        }

        @JavascriptInterface
        public void keepAwake(final boolean on) {
            runOnUiThread(() -> {
                if (on) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            });
        }

        @JavascriptInterface
        public void toast(final String t) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this, t, Toast.LENGTH_SHORT).show());
        }

        @JavascriptInterface
        public String dsStatus() {
            return ds.status();
        }

        @JavascriptInterface
        public void dsInstall() {
            ds.install();
        }

        @JavascriptInterface
        public String dsSubmit(String json) {
            return ds.submit(json);
        }

        @JavascriptInterface
        public String platform() {
            return "android " + Build.VERSION.SDK_INT;
        }
    }
}
