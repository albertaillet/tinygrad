#!/bin/sh
outfile=$1

# clear the output file
if [ -z "$outfile" ]; then
    echo "Usage: $0 <output_file>"
    exit 1
fi
if [ -f "$outfile" ]; then
    rm "$outfile"
fi
ex() {
    echo "➤ $*" >> $outfile
    eval "$*" 2>&1 | tee -a $outfile
}
ex git rev-parse HEAD
ex PYTHONPATH=. METAL=1 STEPS=1 DEBUG=1 python examples/hlb_cifar10.py
