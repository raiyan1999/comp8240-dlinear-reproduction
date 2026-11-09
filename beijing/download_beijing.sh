#!/usr/bin/env bash
# Downloads the UCI Beijing Multi-Site Air Quality data (id 501) into beijing/raw/
set -e
cd "$(dirname "$0")"
mkdir -p raw && cd raw
curl -L -o beijing.zip "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
unzip -o beijing.zip
unzip -o PRSA2017_Data_20130301-20170228.zip
ls PRSA_Data_20130301-20170228
