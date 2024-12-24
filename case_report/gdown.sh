#!/bin/bash

declare -A files
files["1Irf02-QrTNqQYsG0-jNiZw5Iqn0Kk8td"]="input.zip"
files["1OHfOOhkFZBRFZwsLFBx6xrTJWSgqkcho"]="output.zip"


for file_id in "${!files[@]}"; do
  filename=${files[$file_id]}
  gdown --id $file_id -O $filename
  unzip $filename -d extracted_files/
done
