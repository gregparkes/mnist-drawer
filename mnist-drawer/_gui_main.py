
import time
import PySimpleGUI as sg
import numpy as np
import os
# pytorch
import torch
import json

from ._load_data import dataset_mnist_pytorch, loader_mnist_pytorch
from ._model import LeNet
from ._gui_retrain import RetrainWindow
from ._util import rgb_to_hex
#from .recompile import retrain_pyt

class MNISTApplication:

    def __init__(self, verbose: bool):

        self.verbose = verbose

        sg.theme("LightGreen6")
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
        DEF_FONT = "ANY 13"

        self.graph = sg.Graph((600, 600), (0, 0), (GRAPH_DIMS, GRAPH_DIMS),
                     key="-GRAPH-", enable_events=True, drag_submits=True, motion_events=True,
                     background_color="white")
        
        self._numbs = ["zero", "one","two","three","four","five","six","seven","eight","nine"]
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

        # now attempt to load the PyTorch CNN model.
        if os.path.isfile("models/mnist_cnn.pt"):
            DEFAULT_MODEL = os.path.abspath("models/mnist_cnn.pt")
            # load the model and set to loaded = True
            self.model = LeNet().to("cpu")
            model_state = torch.load(DEFAULT_MODEL, map_location=torch.device("cpu"))
            self.model.load_state_dict(model_state)
            self.model_loaded = True
        else:
            DEFAULT_MODEL = ""

        # attempt to load ONNX runtime
        if os.path.isfile("models/ocsvm.json"):
            self.ocsvm_present = True
            from sklearn.linear_model import SGDOneClassSVM
            self._ocsvm = SGDOneClassSVM()
            with open("models/ocsvm.json", "rt") as f:
                ocsvm_state = json.load(f)
                self._ocsvm.coef_ = np.array(ocsvm_state['coef_'])
                self._ocsvm.offset_ = np.array(ocsvm_state['offset_'])
                self._ocsvm.n_iter_ = ocsvm_state['n_iter_']
                self._ocsvm.t_ = ocsvm_state['t_']
                self._ocsvm.n_features_in_ = ocsvm_state['n_features_in_']
        else:
            self.ocsvm_present = False

        layout_l = [
            [sg.Text("Weights", font=DEF_FONT), 
             sg.In(enable_events=True, font=DEF_FONT, key="-FOLDER-", default_text=DEFAULT_MODEL), 
             sg.FileBrowse(font=DEF_FONT)],
            [self.graph],
            [sg.Button("Clear", font=DEF_FONT, key="-CLEAR-", enable_events=True), 
             sg.Button("Random", font=DEF_FONT, key="-SAMPLE-", enable_events=True),
             sg.B("Find Wrong", font=DEF_FONT, key="-WRONG-", enable_events=True),
             sg.B("Retrain", font=DEF_FONT, key="-RETRAIN-", enable_events=True)]
        ]

        layout = [
            [sg.Column(layout_l),
             sg.VSeparator(),
             sg.Column(layout_r)]
        ]

        self.window = sg.Window("MNIST drawing demo", layout, finalize=True, return_keyboard_events=True)

        # fill graph with id rectangles.
        for j in range(self.N):
            for i in range(self.N):
                self.boxloc[i, j, 0] = i * self.BOX_SIZE
                self.boxloc[i, j, 1] = j * self.BOX_SIZE
                self.rects[i, j] = self.graph.draw_rectangle((i * self.BOX_SIZE, j * self.BOX_SIZE),
                    (i * self.BOX_SIZE + self.BOX_SIZE, j * (self.BOX_SIZE) + self.BOX_SIZE),
                    line_color="gray", fill_color="black")
        # bind right click to graph
        self.graph.bind("<ButtonPress-3>", "-RIGHT-CLICK-GRAPH-PRESS-")
        self.graph.bind("<ButtonRelease-3>", "-RIGHT-CLICK-GRAPH-RELEASE-")
        # add circle in the middle.
        self.circle = self.graph.draw_circle((GRAPH_DIMS/2,GRAPH_DIMS/2), self.brush, line_color="red", line_width=3)
        # load MNIST test set.
        _, self.mnist_test = dataset_mnist_pytorch(normalize=False)
        self.test_set_computed = False

    def _compute_incorrect_test_set(self):
        # this must be called once prior to fetching a random wrong sample.
        if self.model_loaded and not self.test_set_computed:
            # calculate predictions.
            _, mnist_test_loader = loader_mnist_pytorch(batch_size=10000)
            # give me the data in one batch
            _test_x, _test_y = next(iter(mnist_test_loader))
            # compute predictions
            preds = self.model(_test_x)
            #print(preds.shape)
            # find the argmax
            _max = torch.argmax(preds, dim=-1)
            # if this is not equal to the test labels, create boolean mask.
            self._incorrect_mask = torch.flatten(torch.argwhere(torch.ne(_test_y, _max)))
            #print(self._incorrect_mask, self._incorrect_mask.shape)
            self.test_set_computed = True

    def _redraw_rect(self, x, y):
        colour = rgb_to_hex(self.grid[x, y])      
        self.graph.tk_canvas.itemconfig(self.rects[x, y], fill=colour)
    
    def _redraw_grid(self):
        for j in range(self.N):
            for i in range(self.N):
                self._redraw_rect(i, j)

    def _redraw_circle(self, mouse):
        self.graph.delete_figure(self.circle)
        self.circle = self.graph.draw_circle(mouse, self.brush, line_color="red", line_width=3)

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
            self._redraw_rect(i, j)
                    
        # compute the time taken.
        t = time.perf_counter()
        # if we've elapsed since the last predict, do it agian.
        if (t - self.time_since_last) > self.model_predict_freq:
            self._predict_label()
            self.time_since_last = t
        # update the mouse
        self.last_mouse = mouse
        # re-draw the circle, we've moved.
        self._redraw_circle(mouse)

    def _predict_label(self, force=False):
        # we need to rotate the thing 90 degrees
        if self.model_loaded or force:
            # set model to evaluation mode
            self.model.eval()
            # load the data from the grid, reshape.
            # clone the grid.
            x = torch.reshape(torch.rot90(torch.tensor(self.grid, dtype=torch.float32)), (1, 1, 28, 28))

            with torch.no_grad():
                # these preds are the log softmax - so 
                preds = torch.exp(torch.flatten(self.model(x)))
                # update the table values in one go.
                self._tab_values = [[a, "{:0.2f}%".format(b*100.)] for a, b in zip(self._numbs, preds.numpy())]
                self.window['-CNNTABLE-'].update(values=self._tab_values)

            # also run in OC-SVM model if present
            if self.ocsvm_present:
                # convert grid into scaled format
                sample = (self.grid * 2 - 1.).reshape(1, 28*28).astype(np.float32)
                # create a decision boundary.
                decision = self._ocsvm.decision_function(sample)
                dec_proportion = (np.sign(decision) * np.sqrt(np.abs(decision) * .2))[0]
                prop_label = "Known" if dec_proportion > 0 else "Unknown"
                self.window['-OCSVMTABLE-'].update(
                    values=[[prop_label, "{:0.2f}".format(dec_proportion)]]
                )

    def mainloop(self):
        # begin the event loop
        self.brush = 25.0
        self.time_since_last = 0
        self.last_mouse = (0, 0)
        right_click_canvas = False

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

                if self.model_loaded:
                    self._predict_label()
                
            elif event == "-FOLDER-":
                # loading a keras model from the folder browser.
                # check values folder
                if os.path.exists(values['-FOLDER-']):
                    self.model = LeNet().to("cpu")
                    model_state = torch.load(values['-FOLDER-'], map_location=torch.device("cpu"))
                    self.model.load_state_dict(model_state)
                    self.model_loaded = True
            
            elif event == "-SAMPLE-":
                test_sample_image, _ = self.mnist_test[np.random.choice(10000)]
                # threshold image into an integer format.
                test_sample_npy = torch.rot90(torch.reshape(test_sample_image, (28, 28)), k=3)
                self.grid = test_sample_npy.numpy()
                # re-draw the grid.
                self._redraw_grid()
                if self.model_loaded:
                    self._predict_label()
            
            elif event == "-WRONG-":
                # computes only once.
                self._compute_incorrect_test_set()
                # now fetch a random wrong index.
                i = np.random.choice(self._incorrect_mask.size()[0])
                wri = int(self._incorrect_mask[i])
                test_image, _ = self.mnist_test[wri]
                # threshold and construct image.
                test_sample_npy = torch.rot90(torch.reshape(test_image, (28, 28)), k=3)
                self.grid = test_sample_npy.numpy()
                # re-draw the grid.
                self._redraw_grid()
                if self.model_loaded:
                    self._predict_label()


            elif event.endswith("+MOVE"):
                # if right click has been pressed, then erase.
                if right_click_canvas:
                    self._draw_tool(values, 0)
                # move the cursor
                self._redraw_circle(mouse)

            elif event == "-GRAPH-+UP":
                # mouse up event. do a prediction.
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
                self._draw_tool(values, 1)
            
            # right click on graph
            elif event == "-GRAPH--RIGHT-CLICK-GRAPH-PRESS-":
                right_click_canvas = True
            elif event == "-GRAPH--RIGHT-CLICK-GRAPH-RELEASE-":
                right_click_canvas = False

            elif event == "-RETRAIN-":
                #self.window.perform_long_operation(lambda :  retrain_pyt(True), "-FINISH_RETRAIN-")
                rt_win = RetrainWindow()
                rt_win.mainloop()
