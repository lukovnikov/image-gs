#!/bin/bash

DIRECTORY="../miniimagenet_256"


# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY does not exist!"
    exit 1
fi

for file in "$DIRECTORY"/*; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        echo "Processing : $(basename "$file")"
        python main.py --input_path="$file" --data_root="" --exp_name="$(basename "$file")" --log_root="results_MIN" --num_gaussians=4000
    fi
done

echo "All files processed!"
