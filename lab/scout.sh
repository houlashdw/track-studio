#!/bin/bash
mkdir -p scout-out dl; cd dl
for r in spicytigermeat/canary_diffsinger spicytigermeat/tiger_diffsinger openvpi/vocoders; do
  n=$(echo $r | tr / _)
  curl -sL https://api.github.com/repos/$r/releases | python3 -c "
import json,sys; R=json.load(sys.stdin)
for x in R[:3]:
  print(x.get('tag_name'), x.get('name'))
  for a in x.get('assets',[]): print('   ', a['size']//1048576, 'MB', a['browser_download_url'])
" > ../scout-out/rel_$n.txt 2>&1
  curl -sL https://raw.githubusercontent.com/$r/HEAD/LICENSE.md -o ../scout-out/lic_$n.md
  curl -sL https://raw.githubusercontent.com/$r/HEAD/README.md -o ../scout-out/readme_$n.md
done
# скачать первый пак Canary и разобрать
U=$(grep -o 'https://[^ ]*\.zip' ../scout-out/rel_spicytigermeat_canary_diffsinger.txt | head -1)
echo "$U" > ../scout-out/canary_url.txt
if [ -n "$U" ]; then curl -sL -o canary.zip "$U"; unzip -q -o canary.zip -d canary; find canary -name '*.zip' -exec unzip -q -o {} -d canary/inner \; ; find canary -type f | xargs ls -la > ../scout-out/canary_files.txt; find canary -name '*.yaml' -o -name '*.txt' -o -name '*.json' | head -40 | while read f; do echo "=== $f"; head -c 3000 "$f"; echo; done > ../scout-out/canary_cfg.txt; fi
