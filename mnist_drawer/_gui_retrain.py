
import PySimpleGUI as sg
# pytorch
from torchinfo import summary

from ._model import Model, ModelParams
from ._load_data import MNISTData
from ._train_model import retrain_pyt
from ._util import parse_float, parse_int


class RetrainWindow:

    def __init__(self, data: MNISTData, params: ModelParams, verbose: bool = False):
        """Defines and initialises a modal window."""
        DEF_TEXT_SIZE = (20, 1)
        DEF_FONT = "ANY 12"

        self.mnist = data
        self.params = params
        self.verbose = verbose

        DEFAULT_BATCH_SIZE = 128
        TOTAL_EPOCHS = 20
        number_of_batches = 60000 // DEFAULT_BATCH_SIZE
        LEARNING_RATE = 0.01
        MOMENTUM = 0.9

        layout = [
            [sg.Text("Model Architecture", font=DEF_FONT, size=DEF_TEXT_SIZE, tooltip="CNN Model architecture"), 
             sg.DropDown(['LeNet'], default_value="LeNet", font=DEF_FONT)],
            [sg.Text("Batch Size", font=DEF_FONT, size=DEF_TEXT_SIZE),
             sg.In(str(DEFAULT_BATCH_SIZE), font=DEF_FONT, key="-BATCH_SIZE-", enable_events=True)],
            [sg.Text("Total Epochs", font=DEF_FONT, size=DEF_TEXT_SIZE),
             sg.In(str(TOTAL_EPOCHS), key="-TOTAL_EPOCHS-", font=DEF_FONT, enable_events=True)],
            [sg.Text("Learning Rate", font=DEF_FONT, size=DEF_TEXT_SIZE, tooltip="SGD learning rate parameter"),
             sg.In(str(LEARNING_RATE), key="-LR-", font=DEF_FONT, enable_events=True)],
            [sg.Text("Momentum", size=DEF_TEXT_SIZE, font=DEF_FONT, tooltip="SGD momentum parameter"),
             sg.In(str(MOMENTUM), key="-MOMENTUM-", font=DEF_FONT, enable_events=True)],
            [sg.Checkbox("Save Model", key="-SAVE_CHECK-", default=True, 
                font=DEF_FONT, size=DEF_TEXT_SIZE, tooltip="Whether to store the model weights")],
            [sg.B("Train", key="-TRAIN-", font=DEF_FONT, enable_events=True), 
             sg.B("Architecture", key="-ARCH-", font=DEF_FONT, enable_events=True)],

            [sg.HSeparator()],

            [sg.Text("Training Progress", size=DEF_TEXT_SIZE, font="ANY 15")],
            [sg.Text(f"Epoch 0/{TOTAL_EPOCHS}", font=DEF_FONT, key="-EPOCH_TEXT-"), 
             sg.ProgressBar(number_of_batches, size=(40, 10), key="-PROG_BAR-")],
            [sg.Text("Time: 0s", font=DEF_FONT, key="-TIME_TEXT-")],
            [sg.Text("Loss: 0.00", font=DEF_FONT, key="-TRAIN_LOSS-")],
        ]

        self.window = sg.Window("MNIST CNN Retrainer", layout, modal=True, finalize=True)
        self.model = Model(self.params)

    def mainloop(self):
        # infinite event loop.
        while True:
            event, values = self.window.read()
            # control+C as in stop program.
            if event in ("Exit", None, sg.WIN_CLOSED, "c:54", "-FINISH_TRAIN-"):
                break

            elif event == "-TRAIN-":
                # block button
                self.window['-TRAIN-'].update(disabled=True)
                # get batch_size 
                self.params.batch_size = parse_int(values['-BATCH_SIZE-'])
                number_of_batches = 60000 // self.params.batch_size
                self.params.n_epochs = parse_int(values['-TOTAL_EPOCHS-'])
                self.params.learning_rate = parse_float(values['-LR-'])
                self.params.momentum = parse_float(values['-MOMENTUM-'])
                # firstly adjust the progress bar length
                self.window['-PROG_BAR-'].update(max=number_of_batches)

                gui_elems = (self.window['-PROG_BAR-'], 
                        self.window['-EPOCH_TEXT-'],
                        self.window['-TIME_TEXT-'],
                        self.window['-TRAIN_LOSS-'])

                self.window.perform_long_operation(
                    lambda : retrain_pyt(
                        gui_elems,
                        self.mnist, 
                        self.params,
                        verbose=self.verbose),
                    "-FINISH_TRAIN-"
                )

            elif event == "-FINISH_TRAIN-":
                # re-load the model
                self.model.is_loaded = False
                self.window.perform_long_operation(self.model.load, "-FINISH_LOAD-")

            elif event == "-FINISH_LOAD-":
                self.model.is_loaded = True

            elif event == "-ARCH-" and self.model.is_loaded:
                # generate pop up with model architecture.
                summary(self.model.net, (self.params.batch_size, 1, self.params.input_size, self.params.input_size))
       
        self.window.close()
