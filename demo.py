from re import X
import time
import PySimpleGUI as sg
import numpy as np
import os
import argparse
import gzip

# disable GPU - we don't need it for predictions. only training.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

#from tensorflow import keras

# test data from MNIST
def load_test_data():

    if os.path.isfile("data/t10k-images-idx3-ubyte.gz"):
        # load local file
        with gzip.open("data/t10k-images-idx3-ubyte.gz","r") as f:
            image_size = 28

            f.read(16)
            buf = f.read(image_size * image_size * 10000)
            test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            test_data = test_data.reshape(10000, image_size, image_size)
            
        with gzip.open('data/t10k-labels-idx1-ubyte.gz','r') as f:
            f.read(8)
            test_labels = np.frombuffer(f.read(10000), dtype=np.uint8)
    else:
        from tensorflow.keras import datasets
        # load from mnist directly
        (_, _), (test_data, test_labels) = datasets.mnist.load_data(path="mnist.npz")
    
    return test_data, test_labels


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

        layout_l = [
            [sg.Text("Keras Model"), sg.In(enable_events=True, key="-FOLDER-"), sg.FolderBrowse()],
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

        # long operation for loading keras
        def _long_op():
            print("importing tensorflow...")
            from tensorflow import keras
            print("loading mnist test data...")
            # load mnist test set
            self.test_X, self.test_y = load_test_data()
            # rotate them 270 degrees
            self.test_X = np.rot90(self.test_X, k=3, axes=[1, 2])
            # load binary example
            print("loading keras model...")
            self.model = keras.models.load_model("models/bin_classifier")
            
        self.window.perform_long_operation(_long_op, "-LOAD_KERAS-")

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
            x = np.rot90(np.copy(self.grid))
            preds = self.model.predict(x.reshape(1, self.N, self.N), verbose=0).flatten()
            # place the values in each GUI element.
            # get the index of the largest
            #idx = np.argmax(preds)
            for i in range(10):
                self.window[f'-PRED{i}-'].update("{:0.1f}%".format(preds[i] * 100))

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

            elif event == "-LOAD_KERAS-":
                self.window['-FOLDER-'].update("models/bin_classifier")
                self.model_loaded = True
                self._predict_label()
                

            elif event == "-FOLDER-":
                # loading a keras model from the folder browser.
                from tensorflow import keras
                self.model = keras.models.load_model(values['-FOLDER-'])

            elif event == "-BRUSH-":
                # changed the brush size.
                self.brush = values['-BRUSH-']
                self.brush_sq = self.brush*self.brush
                self._redraw_circle(mouse)
            
            elif event == "-SAMPLE-":
                # if tensorflow is loaded, extract a sample from MNIST
                if self.model_loaded:
                    # sample an image from test set MNIST
                    self.grid = np.copy(self.test_X[np.random.choice(10000), :, :])
                    # re-populate the grid
                    for j in range(self.N):
                        for i in range(self.N):
                            self._redraw_rect(i, j)
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
