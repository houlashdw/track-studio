#!/bin/bash
# Шифрует mp3 из $1 в $2 открытым ключом server/out_pub.pem (закрытый — только у Claude в сессии).
# Случайный ключ AES оборачивается RSA → $2/key.bin; файлы → *.enc
set -e; S="$1"; O="$2"; mkdir -p "$O"
K=$(openssl rand -hex 32); export K
printf %s "$K" | openssl pkeyutl -encrypt -pubin -inkey "$(dirname "$0")/../server/out_pub.pem" -pkeyopt rsa_padding_mode:oaep -out "$O/key.bin"
find "$S" -name "*.mp3" | while read -r f; do openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -in "$f" -out "$O/$(basename "$f").enc" -pass env:K; done
ls "$O"
