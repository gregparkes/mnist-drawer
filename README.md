# MNIST-drawer

Draw handwritten digits yourself and classify them in *real-time* on the MNIST dataset. We provide a Graphical User Interface (GUI) and a pre-trained neural network model to do the heavy lifting for you.

![Image not found](resources/gui2.png)

Inspired by Sebastian Lague's wonderful YouTube channel and video: https://www.youtube.com/watch?v=hfMk-kjRv4c

## How to use

Clone this repository, load the requirements and call:

```python
python demo.py
```

This should open a GUI for you to play with. Clicking **draw** will paint on white areas, **erase** will remove white areas, the brush slider indicates the radius of the drawing brush, clear will remove the digit and **sample** will retrieve an example from the MNIST dataset. The model will predict whats in the canvas whenever the mouse up event is triggered.

Initially when you run the application, it will take a few seconds to asynchronously import tensorflow, so please be patient.

## Requirements

- Python 3.x
- `tensorflow` 2.x
- `pysimplegui`
- `numpy`

Use the requirements folder with `conda` using: `conda create -f environment.yml`.

## Network architecture

The network takes a binarized version of the MNIST digits (all values are 0 or 1) 

The trained model bears a CNN architecture (written in `keras`) of the following layers:

1. Input layer
2. Conv2D (16 filters, kernel size (3x3)), relu activation
3. MaxPooling2D (2x2)
4. Conv2D (32 filters, kernel size (3x3)), relu activation
5. MaxPooling2D (2x2)
6. Flatten
7. Dense layer (64), relu activation
8. Dropout 20%
9. Dense layer (32), relu activation
10. Final layer (softmax)

The code for which is in `classification.ipynb`. Sometimes the model behaves weirdly, so you can re-train the model yourself using `python recompile.py`. You will want a GPU to speed up model training.
