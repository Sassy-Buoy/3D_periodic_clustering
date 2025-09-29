#!/bin/bash

# Configuration
DATA_DIR="dash_app_viz/center_plots"
BATCH_SIZE=$((200 * 1024 * 1024))  # 200 MB
BRANCH="main"

# Make sure branch is up-to-date
git checkout $BRANCH
git pull origin $BRANCH

# Gather all files
files=($(find "$DATA_DIR" -type f | sort))
batch=()
batch_size=0
batch_num=1

for file in "${files[@]}"; do
    # get file size in bytes
    fsize=$(stat -c%s "$file")

    if (( batch_size + fsize > BATCH_SIZE && batch_size > 0 )); then
        # Commit and push current batch
        echo "Committing batch #$batch_num with $batch_size bytes"
        git add "${batch[@]}"
        git commit -m "Add batch #$batch_num of center_plots"
        git push origin $BRANCH

        # Reset batch
        batch=()
        batch_size=0
        ((batch_num++))
    fi

    # Add current file to batch
    batch+=("$file")
    ((batch_size += fsize))
done

# Commit remaining files
if (( ${#batch[@]} > 0 )); then
    echo "Committing final batch #$batch_num with $batch_size bytes"
    git add "${batch[@]}"
    git commit -m "Add batch #$batch_num of center_plots"
    git push origin $BRANCH
fi

echo "All batches pushed successfully!"
