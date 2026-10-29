#!/bin/bash
# Проверка APK на эмуляторе: установка, запуск, сборка бита, сохранение файлов, трек с голосом.
set -x
mkdir -p ci-out
adb wait-for-device
adb install -r -g TrackStudio.apk 2>&1 | tee ci-out/install.txt
adb shell am start -W -n com.trackstudio.app/.MainActivity 2>&1 | tee ci-out/start.txt
sleep 15
adb exec-out screencap -p > ci-out/1_start.png
PID=""
for i in $(seq 1 30); do PID=$(adb shell pidof com.trackstudio.app | tr -d '\r'); [ -n "$PID" ] && break; sleep 1; done
echo "pid=$PID" | tee ci-out/pid.txt
adb shell cat /proc/net/unix 2>/dev/null | grep -o 'webview_devtools_remote_[0-9]*' | sort -u | tee ci-out/sockets.txt
SOCK=$(head -1 ci-out/sockets.txt)
[ -z "$SOCK" ] && SOCK="webview_devtools_remote_$PID"
adb forward tcp:9222 "localabstract:$SOCK"
timeout 3900 python3 ci/devtools_test.py 2>&1 | tee ci-out/test.log
adb exec-out screencap -p > ci-out/9_end.png
adb shell ls -la /sdcard/Download/TrackStudio > ci-out/downloads.txt 2>&1
adb pull /sdcard/Download/TrackStudio ci-out/downloads 2>&1 | tail -2
adb logcat -d > ci-out/logcat-full.txt 2>&1
grep -E "chromium|AndroidRuntime|FATAL|TrackStudio" ci-out/logcat-full.txt | tail -400 > ci-out/logcat.txt
rm -f ci-out/logcat-full.txt
exit 0
