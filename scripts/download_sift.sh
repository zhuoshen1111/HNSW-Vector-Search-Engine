#!/usr/bin/env bash
# Downloads the SIFT1M benchmark (TEXMEX corpus, ~161 MB compressed) into data/sift/.
#   base: 1,000,000 x 128 float vectors   query: 10,000   ground truth: 100 nearest ids per query
# Source: http://corpus-texmex.irisa.fr/ (Jegou, Douze, Schmid, "Product quantization for nearest
# neighbor search", TPAMI 2011).
#
#   scripts/download_sift.sh              full archive (~570 MB extracted) for `--only sift`
#   scripts/download_sift.sh --base-100k  space-saving: streams the archive and keeps only the first
#                                         100,000 base vectors (51.6 MB) plus queries and ground
#                                         truth; enough for `--only sift_delete`
set -euo pipefail
cd "$(dirname "$0")/.."
URL=ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
mkdir -p data/sift
if [ "${1:-}" = "--base-100k" ]; then
  # Each fvecs row is 4 + 128*4 = 516 bytes; head closes the pipe after 100K rows (a "Broken pipe"
  # message from tar/curl is expected), so most of the archive is never downloaded.
  set +o pipefail
  curl -fsL --retry 3 "$URL" | tar -xzOf - sift/sift_base.fvecs | head -c 51600000 > data/sift/sift_base_100k.fvecs
  set -o pipefail
  for f in sift_query.fvecs sift_groundtruth.ivecs; do
    [ -f "data/sift/$f" ] || curl -fsL --retry 3 "$URL" | tar -xzf - -C data "sift/$f"
  done
  ls -l data/sift
  exit 0
fi
if [ -f data/sift/sift_base.fvecs ]; then
  echo "data/sift already present"
  exit 0
fi
curl -fL --retry 3 -o data/sift.tar.gz "$URL"
tar -xzf data/sift.tar.gz -C data
rm -f data/sift.tar.gz
ls -l data/sift
