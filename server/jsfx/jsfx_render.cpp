// jsfx_render: офлайн-цепочка JSFX-эффектов (тех же, что в Reaper) без Reaper — на библиотеке ysfx (Apache-2.0)
// jsfx_render <sr> <in.f32> <out.f32> "<file.jsfx>[@<слайдер>=<значение>,...]" ...
// аудио: сырой float32 стерео (interleaved); слайдеры — номер как в файле (slider1 = 1)
#include "ysfx.h"
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void logr(intptr_t, ysfx_log_level lv, const char *m) { if (lv != ysfx_log_info) fprintf(stderr, "[%s] %s\n", ysfx_log_level_string(lv), m); }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: jsfx_render <sr> <in.f32> <out.f32> <fx spec>...\n"); return 2; }
    double sr = atof(argv[1]);
    FILE *f = fopen(argv[2], "rb"); if (!f) { perror("in"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f) / (sizeof(float) * 2); fseek(f, 0, SEEK_SET);
    std::vector<float> buf(n * 2); if (fread(buf.data(), sizeof(float), n * 2, f) != (size_t)(n * 2)) { fprintf(stderr, "short read\n"); return 1; } fclose(f);
    const uint32_t B = 512;
    for (int a = 4; a < argc; ++a) {
        std::string spec = argv[a], path = spec, sl; size_t at = spec.find('@');
        if (at != std::string::npos) { path = spec.substr(0, at); sl = spec.substr(at + 1); }
        ysfx_config_t *cfg = ysfx_config_new(); ysfx_set_log_reporter(cfg, &logr); ysfx_register_builtin_audio_formats(cfg); ysfx_guess_file_roots(cfg, path.c_str());
        ysfx_t *fx = ysfx_new(cfg); ysfx_config_free(cfg);
        if (!ysfx_load_file(fx, path.c_str(), 0) || !ysfx_compile(fx, ysfx_compile_no_gfx)) { fprintf(stderr, "не загрузился: %s\n", path.c_str()); ysfx_free(fx); continue; }
        ysfx_set_sample_rate(fx, sr); ysfx_set_block_size(fx, B); ysfx_init(fx);
        // слайдеры: "2=-6,4=0.5"
        size_t p = 0;
        while (p < sl.size()) {
            size_t c = sl.find(',', p); std::string kv = sl.substr(p, c == std::string::npos ? std::string::npos : c - p); p = c == std::string::npos ? sl.size() : c + 1;
            size_t e = kv.find('='); if (e == std::string::npos) continue;
            uint32_t idx = (uint32_t)atoi(kv.substr(0, e).c_str()) - 1; double v = atof(kv.substr(e + 1).c_str());
            if (ysfx_slider_exists(fx, idx)) ysfx_slider_set_value(fx, idx, v, true); else fprintf(stderr, "нет слайдера %u в %s\n", idx + 1, path.c_str());
        }
        uint32_t ni = ysfx_get_num_inputs(fx), no = ysfx_get_num_outputs(fx); if (ni < 2) ni = 2; if (no < 2) no = 2;
        std::vector<std::vector<float>> I(ni, std::vector<float>(B)), O(no, std::vector<float>(B));
        std::vector<const float *> ip(ni); std::vector<float *> op(no);
        for (uint32_t c = 0; c < ni; ++c) ip[c] = I[c].data(); for (uint32_t c = 0; c < no; ++c) op[c] = O[c].data();
        ysfx_time_info_t ti{}; ti.tempo = 120; ti.playback_state = ysfx_playback_playing; ti.time_signature[0] = 4; ti.time_signature[1] = 4;
        double pdc = ysfx_get_pdc_delay(fx); long lat = (long)(pdc > 0 ? pdc : 0), total = n + lat, w = 0;
        std::vector<float> out(n * 2, 0.f);
        for (long s = 0; s < total; s += B) {
            uint32_t k = (uint32_t)((total - s) < (long)B ? (total - s) : B);
            for (uint32_t i = 0; i < k; ++i) { long j = s + i; float L = j < n ? buf[j * 2] : 0.f, R = j < n ? buf[j * 2 + 1] : 0.f; I[0][i] = L; I[1][i] = R; for (uint32_t c = 2; c < ni; ++c) I[c][i] = 0; }
            ti.time_position = s / sr; ti.beat_position = ti.time_position * ti.tempo / 60.0; ysfx_set_time_info(fx, &ti);
            ysfx_process_float(fx, ip.data(), op.data(), ni, no, k);
            for (uint32_t i = 0; i < k; ++i) { long j = s + i - lat; if (j >= 0 && j < n) { out[j * 2] = O[0][i]; out[j * 2 + 1] = O[1][i]; } }
            w += k;
        }
        fprintf(stderr, "ok: %s (%s), задержка %ld\n", path.c_str(), ysfx_get_name(fx), lat);
        buf.swap(out); ysfx_free(fx);
    }
    FILE *g = fopen(argv[3], "wb"); fwrite(buf.data(), sizeof(float), n * 2, g); fclose(g);
    return 0;
}
