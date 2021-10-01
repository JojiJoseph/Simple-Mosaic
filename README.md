# Simple Mosaic

A simple python script to generate photographic mosaic.


To use,

```
python3 simple_mosaic.py
```

## Working

This program compares each 8x8 patch of the input image to images in the CIFAR-10 dataset. Then it replaces each patch with the closest match in CIFAR-10.
