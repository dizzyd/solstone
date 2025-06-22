#!/bin/bash

# Run gemini_mic.py and transcribe.py in parallel with restart loops

mkdir -p logs

# Function to run gemini_mic.py in a restart loop
run_gemini_mic() {
  while true; do
    start_ts=$(date +%Y%m%d_%H%M%S)
    echo "Starting gemini_mic.py at $start_ts"
    python3 "$(dirname "$0")/hear/gemini_mic.py" "$@"
    echo "gemini_mic.py exited, restarting in 1 second..."
    sleep 1
  done
}

# Function to run transcribe.py in a restart loop
run_transcribe() {
  # Extract the save directory from arguments (first positional arg)
  save_dir="${1:-$(pwd)}"
  while true; do
    start_ts=$(date +%Y%m%d_%H%M%S)
    echo "Starting transcribe.py at $start_ts"
    python3 "$(dirname "$0")/hear/transcribe.py" "$save_dir"
    echo "transcribe.py exited, restarting in 1 second..."
    sleep 1
  done
}

# Start both processes in background
run_gemini_mic "$@" &
GEMINI_PID=$!

run_transcribe "$@" &
TRANSCRIBE_PID=$!

# Function to cleanup background processes
cleanup() {
  echo "Stopping processes..."
  kill $GEMINI_PID $TRANSCRIBE_PID 2>/dev/null
  wait
  exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "Started gemini_mic.py (PID: $GEMINI_PID) and transcribe.py (PID: $TRANSCRIBE_PID)"
echo "Press Ctrl+C to stop both processes"

# Wait for both processes
wait

