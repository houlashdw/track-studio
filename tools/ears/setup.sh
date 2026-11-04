#!/bin/bash
# «Уши» для проверки звука: DNSMOS + NISQA + Audiobox Aesthetics (+ свои замеры). Ставит всё в /home/claude/ears
set -e
E=/home/claude/ears; mkdir -p $E/pkgs $E/nisqa/weights $E/nisqa/nisqa $E/aesm
P="--proxy ${HTTPS_PROXY:-http://127.0.0.1:36773}"
NO_PROXY= no_proxy= pip download speechmos audiobox_aesthetics --no-deps -d $E/pkgs $P
(cd $E && rm -rf sm && mkdir sm && cd sm && unzip -o -q ../pkgs/speechmos-*.whl)
for f in weights/nisqa.tar weights/nisqa_tts.tar nisqa/NISQA_lib.py nisqa/NISQA_model.py run_predict.py; do curl -sSL -o $E/nisqa/$f https://raw.githubusercontent.com/gabrielmittag/NISQA/master/$f; done
curl -sSL -o $E/aesm/checkpoint.pt https://github.com/houlashdw/track-studio/releases/download/ears-models/checkpoint.pt
cp "$(dirname "$0")/ears.py" $E/
echo "python: нужны torch, torchaudio, librosa, onnxruntime, pandas, matplotlib, soundfile; затем pip install --no-deps $E/pkgs/audiobox_aesthetics-*.whl"
