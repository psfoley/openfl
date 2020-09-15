#!/bin/bash
set -e

MEM_USE=($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits))

free=-1
i=0

for mem in "${MEM_USE[@]}"
do
    # Update min if applicable
    if [[ "$mem" -lt 2 ]]; then
        free=$i
        break
    fi
    let i=$i+1
done
echo $free
