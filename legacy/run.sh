#!/bin/bash
set -e

echo "=== EgoDex → LeRobot Pipeline ==="
echo "Starting at $(date)"

# Download EgoDex test set (16 GB)
DATA_DIR="/data/egodex"
OUTPUT_DIR="/data/output"
mkdir -p $DATA_DIR $OUTPUT_DIR

if [ ! -d "$DATA_DIR/test" ]; then
    echo "Downloading EgoDex test set (16 GB)..."
    curl -L "https://ml-site.cdn-apple.com/datasets/egodex/test.zip" -o /tmp/test.zip
    echo "Extracting..."
    unzip -q /tmp/test.zip -d $DATA_DIR
    rm /tmp/test.zip
    echo "Download complete."
else
    echo "EgoDex test set already exists."
fi

echo ""
echo "Processing EgoDex → LeRobot format..."
python3 /workspace/egodex_to_lerobot.py \
    --input $DATA_DIR/test \
    --output $OUTPUT_DIR/lerobot_egodex_sample \
    --max-episodes 50 \
    --fps 10

echo ""
echo "=== Pipeline complete ==="
echo "Output: $OUTPUT_DIR/lerobot_egodex_sample"
echo "Finished at $(date)"

# Keep pod alive for inspection
echo "Pod staying alive for 30min for inspection..."
sleep 1800
