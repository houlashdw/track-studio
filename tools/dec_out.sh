#!/bin/bash
# tools/dec_out.sh <branch> <outdir> — забрать ветку с зашифрованными результатами и расшифровать закрытым ключом сессии
set -e; B="$1"; O="$2"; T=$(mktemp -d); mkdir -p "$O"
R=$(git rev-parse --show-toplevel)
git -C "$R" fetch -q origin "$B"
git -C "$R" archive "origin/$B" | tar -x -C "$T"
K=$(openssl pkeyutl -decrypt -inkey /root/.ts_out.pem -pkeyopt rsa_padding_mode:oaep -in "$T/key.bin"); export K
for f in "$T"/*.enc; do n=$(basename "$f" .enc); openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -in "$f" -out "$O/$n" -pass env:K; echo "$n"; done
rm -rf "$T"
