
import time
import PySimpleGUI as sg
import numpy as np
import os
import argparse
import gzip
# pytorch
import torch

from ._load_data import dataset_mnist_pytorch
from ._model import LeNet

# disable GPU - we don't need it for predictions. only training.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

#from tensorflow import keras

# test data from MNIST
def load_test_data():

    if os.path.isfile("../data/t10k-images-idx3-ubyte.gz"):
        # load local file
        with gzip.open("../data/t10k-images-idx3-ubyte.gz","r") as f:
            image_size = 28

            f.read(16)
            buf = f.read(image_size * image_size * 10000)
            test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            test_data = test_data.reshape(10000, image_size, image_size)
            
        with gzip.open('../data/t10k-labels-idx1-ubyte.gz','r') as f:
            f.read(8)
            test_labels = np.frombuffer(f.read(10000), dtype=np.uint8)
    else:
        raise ValueError("dataset not present.")
    
    return test_data, test_labels


def rgb_to_hex(c):
    # c in the range [0..1]
    a = int(c * 255)
    return "#%02x%02x%02x" % (np.clip(a + 30, 0, 255), a, a)


class MNISTApplication:

    def __init__(self, verbose: bool):

        self.verbose = verbose

        sg.theme("Dark Blue 3")
        self.N = 28
        self.grid = np.zeros((self.N, self.N), dtype="i")
        self.boxloc = np.zeros((self.N, self.N, 2), dtype="i")
        self.rects = np.zeros_like(self.grid)

        self.BOX_SIZE = 20
        self.brush = 25
        self.brush_sq = self.brush*self.brush
        self.model = None
        self.model_loaded = False
        # in seconds
        self.model_predict_freq = 0.4
        GRAPH_DIMS = 560

        self.graph = sg.Graph((600, 600), (0, 0), (GRAPH_DIMS, GRAPH_DIMS),
                     key="-GRAPH-", enable_events=True, drag_submits=True,motion_events=True,
                     background_color="white")
        
        layout_r = [
            [sg.Text("MNIST Predictor", font="ANY 20")],
            *[[sg.Text(F"Predicted '{i}': ", font="ANY 13"), sg.Text("0.0%", key=f"-PRED{i}-", font="ANY 13")] for i in range(10)]
        ]

        BRUSH_MIN = 15
        BRUSH_MAX = 45
        BRUSH_INCREMENT = 0.5
        if os.path.isfile("models/mnist_cnn.pt"):
            DEFAULT_MODEL = os.path.abspath("models/mnist_cnn.pt")
            # load the model and set to loaded = True
            self.model = LeNet().to("cpu")
            model_state = torch.load(DEFAULT_MODEL, map_location=torch.device("cpu"))
            self.model.load_state_dict(model_state)
            self.model_loaded = True
        else:
            DEFAULT_MODEL = ""

        layout_l = [
            [sg.Text("Torch Model"), sg.In(enable_events=True, key="-FOLDER-", default_text=DEFAULT_MODEL), sg.FileBrowse()],
            [self.graph],
            [sg.Radio("Draw", "drawing", key="-DRAW-", default=True), sg.Radio("Erase", "drawing", key="-ERASE-"),
            sg.Text("Brush:"), sg.Slider((BRUSH_MIN, BRUSH_MAX), self.brush, BRUSH_INCREMENT, orientation="h", key="-BRUSH-", enable_events=True),
            sg.Button("Clear", key="-CLEAR-", enable_events=True), 
            sg.Button("Sample", key="-SAMPLE-", enable_events=True)]
        ]

        layout = [
            [sg.Column(layout_l),
            sg.VSeparator(),
            sg.Column(layout_r)]
        ]

        self.window = sg.Window("MNIST drawing demo", layout, finalize=True)

        # fill graph with id rectangles.
        for j in range(self.N):
            for i in range(self.N):
                self.boxloc[i, j, 0] = i * self.BOX_SIZE
                self.boxloc[i, j, 1] = j * self.BOX_SIZE
                self.rects[i, j] = self.graph.draw_rectangle((i * self.BOX_SIZE, j * self.BOX_SIZE),
                    (i * self.BOX_SIZE + self.BOX_SIZE, j * (self.BOX_SIZE) + self.BOX_SIZE),
                    line_color="gray", fill_color="black")
        
        # add circle in the middle.
        self.circle = self.graph.draw_circle((GRAPH_DIMS/2,GRAPH_DIMS/2), self.brush, line_color="red", line_width=3)

        # load MNIST test set.
        _, self.mnist_test = dataset_mnist_pytorch(normalize=False)

    def _redraw_rect(self, x, y):
        # determine colour from the grid.
        colour_value = self.grid[x, y]
        colour = "black" if colour_value == 0 else "white"

        self.graph.delete_figure(self.rects[x, y])
        self.rects[x, y] = self.graph.draw_rectangle(
            (self.boxloc[x, y, 0], self.boxloc[x, y, 1]),
            (self.boxloc[x, y, 0]+self.BOX_SIZE, self.boxloc[x, y, 1]+self.BOX_SIZE),
            line_color="gray", fill_color=colour
        )
    
    def _redraw_circle(self, mouse):
        #self.graph.move_figure(self.circle, *delta)
        self.graph.delete_figure(self.circle)
        self.circle = self.graph.draw_circle(mouse, self.brush, line_color="red", line_width=3)

    def _predict_label(self, force=False):
        # we need to rotate the thing 90 degrees
        if self.model_loaded or force:
            # load the data from the grid, reshape.
            # clone the grid.
            x = torch.reshape(torch.rot90(torch.tensor(self.grid, dtype=torch.float32)), (1, 1, 28, 28))



            with torch.no_grad():
                # these preds are the log softmax - so 
                preds = torch.exp(torch.flatten(self.model(x)))
                for i in range(10):
                    # convert number [0..1] into hex 
                    col = rgb_to_hex(preds[i] / 2. + 0.5)
                    self.window[f'-PRED{i}-'].update(value="{:0.1f}%".format(preds[i] * 100), text_color=col)
                # set the one with the highest value to a different colour.


    def mainloop(self):
        # begin the event loop
        time_since_last = 0
        last_mouse = (0, 0)

        while True:
            
            event,values = self.window.read()
            #print(event)
            if event in (sg.WIN_CLOSED, None, "Exit"):
                break

            mouse = values['-GRAPH-']
            if self.verbose:
                print(event)

            if event == "-CLEAR-":
                # clears the canvas.
                self.grid.fill(0)
                # loop through all the rectangles and re-draw them
                for j in range(self.N):
                    for i in range(self.N):
                        self._redraw_rect(i, j)
                
            elif event == "-FOLDER-":
                # loading a keras model from the folder browser.
                #from tensorflow import keras
                self.model = LeNet().to("cpu")
                model_state = torch.load(values['-FOLDER-'], map_location=torch.device("cpu"))
                self.model.load_state_dict(model_state)
                #self.model = keras.models.load_model(values['-FOLDER-'])
                self.model_loaded = True

            elif event == "-BRUSH-":
                # changed the brush size.
                self.brush = values['-BRUSH-']
                self.brush_sq = self.brush*self.brush
                self._redraw_circle(mouse)
            
            elif event == "-SAMPLE-":
                # if tensorflow is loaded, extract a sample from MNIST
                # sample an image from test set MNIST
                #self.grid = np.copy(self.test_X[np.random.choice(10000), :, :])
                # select a sample
                test_sample_image, _ = self.mnist_test[np.random.choice(10000)]
                # threshold image into an integer format.
                test_sample_npy = torch.where(torch.rot90(torch.reshape(test_sample_image, (28, 28)), k=3) > 0., 1, 0)
                #print(test_sample_npy)
                #self.grid = np.rot90(np.where((test_sample_npy) > 0., 1., 0.), k=3)
                # re-populate the grid
                self.grid = test_sample_npy.numpy()
                for j in range(self.N):
                    for i in range(self.N):
                        self._redraw_rect(i, j)
                if self.model_loaded:
                    self._predict_label()
            
            elif event.endswith("+MOVE"):
                # move the cursor
                self._redraw_circle(mouse)

            elif event.endswith("+UP"):
                # mouse up event. do a prediction.
                self._predict_label()

            elif event == "-GRAPH-":
                if mouse == (None, None) or mouse == last_mouse:
                    continue
                #print(mouse)
                value = 1 if values['-DRAW-'] else 0

                # vectorize the condition matrix
                condition = (np.square(self.boxloc[:, :, 0] - mouse[0]) + np.square(self.boxloc[:, :, 1] - mouse[1])) <= self.brush_sq
                # use the condition matrix to vectorize-set the grid
                self.grid[condition] = value
                # loop over the grid
                x, y = np.where(condition)
                # zip the points togther, loop over and re-draw the corresponding rectangles.
                for i, j in zip(x, y):
                    self._redraw_rect(i, j)
                            
                # compute the time taken.
                t = time.perf_counter()
                # if we've elapsed since the last predict, do it agian.
                if (t - time_since_last) > self.model_predict_freq:
                    self._predict_label()
                    time_since_last = t
                # update the mouse
                last_mouse = mouse
                # re-draw the circle, we've moved.
                self._redraw_circle(mouse)


def main():
    parser = argparse.ArgumentParser(description='MNIST drawing demo.')
    parser.add_argument("-v", "--verbose",  action="store_true", default=False)
    args = parser.parse_args()

    # start the app
    app = MNISTApplication(verbose=args.verbose)#
    # begin main loop
    app.mainloop()
    # if we leave the loop, always close the application.
    app.window.close()


if __name__ == "__main__":
    main()
