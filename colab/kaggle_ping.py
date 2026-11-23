import subprocess, time
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout, flush=True)
open("/kaggle/working/ok.txt", "w").write("ok")
