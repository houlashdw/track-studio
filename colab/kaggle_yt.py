"""Kaggle: биты free for profit с YouTube → /kaggle/working/beats (выход ноутбука = датасет для дообучения)."""
import os, subprocess, sys
W = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('yt_out')
subprocess.run(f'{sys.executable} -m pip install -q -U yt-dlp', shell=True)
subprocess.run(f'wget -q -O /tmp/grab.py https://raw.githubusercontent.com/houlashdw/track-studio/main/tools/yt/grab.py', shell=True, check=True)
subprocess.run([sys.executable, '/tmp/grab.py', W + '/beats', os.environ.get('N', '2')])
