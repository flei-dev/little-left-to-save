#!/usr/bin/env bash
# Full pipeline.  Usage: bash run.sh cpu2 | cpu1 | status | zip | stop | check1 | check2
#   cpu2: E1, E2, E3, E5      cpu1: AV1 encoding, then E4
# Stages with a DONE marker in results/<stage>/ are skipped; move results/ away to re-run everything.
# ---------- only cpu1 (AV1 encoding) uses the next three paths ----------
export VIDEOS=~/TMM/videos
export FFMPEG=~/TMM/ffmpeg/bin/ffmpeg
export FFPROBE=~/TMM/ffmpeg/bin/ffprobe
# -------------------------------------------------------------------------
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$1" in
  cpu1|cpu2)
    python3 -m cstudy.check "$1" | tee -a "run_$1.log" || exit 1
    setsid nohup python3 -m cstudy.main "$1" >> "run_$1.log" 2>&1 < /dev/null &
    echo "Started in the background. Progress: bash run.sh status" ;;
  status)
    python3 -m cstudy.main status
    for f in run_cpu*.log; do [ -f "$f" ] && { echo "--- last 3 lines of $f"; tail -n 3 "$f"; }; done ;;
  zip)    python3 -m cstudy.main zip ;;
  check1) python3 -m cstudy.check cpu1 ;;
  check2) python3 -m cstudy.check cpu2 ;;
  stop)   pkill -f "cstudy.main" ; pkill -f "cstudy.run" ; pkill -f "run_all.py" ; pkill -f encode_vmaf.py ; echo stopped ;;
  *) echo "Usage: bash run.sh cpu2 | cpu1 | status | zip | stop" ;;
esac
