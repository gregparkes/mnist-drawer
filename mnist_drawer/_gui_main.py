
import time
import PySimpleGUI as sg
import numpy as np
import os
# pytorch
import torch
import torchvision.transforms as T
from scipy.signal import convolve2d

from ._load_data import MNISTData
from ._model import Model, ModelParams
from ._gui_retrain import RetrainWindow
from ._util import rgb_to_hex

class MNISTApplication:

    def __init__(self, verbose: bool):

        self.verbose = verbose

        sg.theme("LightGreen6")
        # specify a 28x28 grid of cells to draw into.
        self.N = 28
        # holds the actual colour
        self.grid = np.zeros((self.N, self.N), dtype="i")
        # [x, y] indices of each grid mapping
        self.boxloc = np.zeros((self.N, self.N, 2), dtype="i")
        # an array holding the pysimplegui reference pointers
        self.rects = np.zeros_like(self.grid)

        self.BOX_SIZE = 20
        self.brush = 25
        self.brush_sq = self.brush*self.brush
        # in seconds
        self.model_predict_freq = 0.4
        GRAPH_DIMS = 560
        DEF_FONT = "ANY 13"

        # model arguments
        DEFAULT_MODEL = os.path.abspath("models/mnist_cnn.pt")
        self.model_params = ModelParams(128)
        self.model = Model(self.model_params)

        # raw dataset
        self.mnist = MNISTData(self.model_params)

        # GUI starts here
        self.graph = sg.Graph((600, 600), (0, 0), 
                    (GRAPH_DIMS, GRAPH_DIMS),
                     key="-GRAPH-", enable_events=True, 
                     drag_submits=True, 
                     motion_events=True,
                     background_color="white")
        
        self._numbs = ["zero", "one","two","three","four",
            "five","six","seven","eight","nine"]
        vals = ["0.00%" for _ in range(10)]
        self._tab_values = [[a, b] for a, b in zip(self._numbs, vals)]

        tab1 = sg.Table(
            values=self._tab_values,
            headings=['Number', 'Score'],
            max_col_width=35,
            auto_size_columns=False,
            display_row_numbers=False,
            justification="right",
            num_rows=10,
            font=DEF_FONT,
            key="-CNNTABLE-",
            tooltip="CNN model accuracies"
        )

        tab2 = sg.Table(
            values=[["Known", "0.00%"]],
            headings=['State', 'Score'],
            max_col_width=35,
            auto_size_columns=False,
            display_row_numbers=False,
            justification="right",
            num_rows=1,
            font=DEF_FONT,
            key="-OCSVMTABLE-",
            tooltip="One-Class SVM model accuracies"
        )

        layout_r = [
            [sg.Text("MNIST Predictor", font="ANY 15")],
             [tab1],

             [sg.HSeparator()],
            
            [sg.Text("OC-SVM Predictor", font="ANY 15")],
            [tab2]
        ]

        layout_l = [
            [sg.Text("Weights", font=DEF_FONT), 
             sg.In(enable_events=True, font=DEF_FONT, key="-FOLDER-", default_text=DEFAULT_MODEL), 
             sg.FileBrowse(font=DEF_FONT)],
            [self.graph],
            [sg.B("Clear", font=DEF_FONT, key="-CLEAR-", enable_events=True), 
             sg.B("Random", font=DEF_FONT, key="-SAMPLE-", enable_events=True),
             sg.B("Convolve", font=DEF_FONT, key="-CONVOLVE-", enable_events=True),
             sg.B("Retrain", font=DEF_FONT, key="-RETRAIN-", enable_events=True)]
        ]

        layout = [
            [sg.Column(layout_l),
             sg.VSeparator(),
             sg.Column(layout_r)]
        ]

        self.window = sg.Window("MNIST drawing demo", layout, 
            finalize=True, return_keyboard_events=True)

        # fill graph with id rectangles.
        for j in range(self.N):
            for i in range(self.N):
                self.boxloc[i, j, 0] = i * self.BOX_SIZE
                self.boxloc[i, j, 1] = j * self.BOX_SIZE
                self.rects[i, j] = self.graph.draw_rectangle(
                    (i * self.BOX_SIZE, j * self.BOX_SIZE),
                    (i * self.BOX_SIZE + self.BOX_SIZE, j * (self.BOX_SIZE) + self.BOX_SIZE),
                    line_color="gray", fill_color="black")
        # bind right click to graph
        self.graph.bind("<ButtonPress-3>", "-RIGHT-CLICK-GRAPH-PRESS-")
        self.graph.bind("<ButtonRelease-3>", "-RIGHT-CLICK-GRAPH-RELEASE-")
        # add circle in the middle.
        self.circle = self.graph.draw_circle((GRAPH_DIMS/2,GRAPH_DIMS/2),
            self.brush, line_color="red", line_width=3)

    def _redraw_rect(self, x: int, y: int):
        colour = rgb_to_hex(self.grid[x, y])      
        self.graph.tk_canvas.itemconfig(self.rects[x, y], fill=colour)
    
    def _redraw_grid(self):
        for j in range(self.N):
            for i in range(self.N):
                self._redraw_rect(i, j)

    def _redraw_circle(self, mouse: tuple[int, int]):
        self.graph.delete_figure(self.circle)
        self.circle = self.graph.draw_circle(mouse, self.brush, 
            line_color="red", line_width=3)

    def _draw_tool(self, values, colour):
        mouse = values['-GRAPH-']
        if mouse == (None, None) or mouse == self.last_mouse:
            return
        # vectorize the condition matrix
        condition = (np.square(self.boxloc[:, :, 0] - mouse[0]) + np.square(self.boxloc[:, :, 1] - mouse[1])) <= self.brush_sq
        # use the condition matrix to vectorize-set the grid
        self.grid[condition] = colour
        # loop over the grid
        x, y = np.where(condition)
        # zip the points togther, loop over and re-draw the corresponding rectangles.
        for i, j in zip(x, y):
            # set rect[i, j] on the canvas to the 'self.grid' colour
            self._redraw_rect(i, j)
                    
        # compute the time taken.
        t = time.perf_counter()
        # if we've elapsed since the last predict, do it agian.
        if self.model.is_loaded and ((t - self.time_since_last) > self.model_predict_freq):
            self._predict_label()
            self.time_since_last = t
        # update the mouse
        self.last_mouse = mouse
        # re-draw the circle, we've moved.
        self._redraw_circle(mouse)

    def _predict_label(self, force=False):
        # we need to rotate the thing 90 degrees
        if self.model.is_loaded or force:
            y = self.model.forward(self.grid)
            # update the table values in one go.
            self._tab_values = [[a, "{:0.2f}%".format(b*100.)] for a, b in zip(self._numbs, y)]
            self.window['-CNNTABLE-'].update(values=self._tab_values)

    def mainloop(self):
        # begin the event loop
        self.brush = 25.0
        self.time_since_last = 0
        self.last_mouse = (0, 0)
        right_click_canvas = False

        self.window.perform_long_operation(self.mnist.load, "-DATA_LOADED-")
        # load model
        self.window.perform_long_operation(self.model.load, "-MODEL_LOADED-")

        gkern2d = np.random.uniform(0.85, 0.95, size=(3, 3))
        gkern2d[np.diag_indices_from(gkern2d)] = 1
        gkern2d /= np.sum(gkern2d)

        while True:
            
            event,values = self.window.read()
            #print(event)
            if event in (sg.WIN_CLOSED, None, "Exit", "c:54"):
                break

            mouse = values['-GRAPH-']
            if self.verbose:
                print(event)

            if event == "-CLEAR-":
                # clears the canvas.
                self.grid.fill(0)
                # loop through all the rectangles and re-draw them
                self._redraw_grid()

                if self.model.is_loaded:
                    self._predict_label()

            elif event == "-MODEL_LOADED-" and self.model.net:
                self.model.is_loaded = True
            
            elif event == "-DATA_LOADED-" and self.mnist.test_set:
                self.mnist.is_loaded = True

            elif event == "-FOLDER-":
                # loading a keras model from the folder browser.
                # check values folder
                if os.path.exists(values['-FOLDER-']):
                    self.model.is_loaded = False
                    self.window.perform_long_operation(self.model.load, "-MODEL_LOADED-")
            
            elif event == "-SAMPLE-" and self.mnist.is_loaded:
                test_sample_image, _ = self.mnist.test_set[np.random.choice(10000)]
                # threshold image into an integer format.
                test_sample_npy = torch.rot90(
                    torch.reshape(test_sample_image, (28, 28)), k=3)
                self.grid = test_sample_npy.numpy()
                # re-draw the grid.
                self._redraw_grid()
                if self.model.is_loaded:
                    self._predict_label()
            
            elif event == "-CONVOLVE-":
                # perform convolution on the grid
                self.grid = convolve2d(self.grid, gkern2d, mode="same")
                # loop through all the rectangles and re-draw them
                self._redraw_grid()
                if self.model.is_loaded:
                    self._predict_label()

            elif event.endswith("+MOVE"):
                # if right click has been pressed, then erase.
                if right_click_canvas:
                    self._draw_tool(values, 0)
                # move the cursor
                self._redraw_circle(mouse)

            elif event == "-GRAPH-+UP":
                # mouse up event. do a prediction.
                if self.model.is_loaded:
                    self._predict_label()

            elif event == "MouseWheel:Up":
                # make brush larger
                self.brush = np.clip(self.brush + 2., 15., 45.)
                self.brush_sq = self.brush*self.brush
                self._redraw_circle(mouse)

            elif event == "MouseWheel:Down":
                # make brush smaller
                self.brush = np.clip(self.brush - 2., 15., 45.)
                self.brush_sq = self.brush*self.brush
                self._redraw_circle(mouse)

            # = LEFT-CLICK
            elif event == "-GRAPH-":
                """Add white to the surrounding boxes."""
                self._draw_tool(values, 1)
            
            # right click on graph
            elif event == "-GRAPH--RIGHT-CLICK-GRAPH-PRESS-":
                right_click_canvas = True
            elif event == "-GRAPH--RIGHT-CLICK-GRAPH-RELEASE-":
                right_click_canvas = False

            elif event == "-RETRAIN-":
                #self.window.perform_long_operation(lambda :  retrain_pyt(True), "-FINISH_RETRAIN-")
                rt_win = RetrainWindow(self.mnist, self.model_params, self.verbose)
                rt_win.mainloop()
