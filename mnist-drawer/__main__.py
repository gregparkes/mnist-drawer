"""Command line interface of mnist drawer."""

import argparse

from ._gui_main import MNISTApplication
from .recompile import retrain_pyt, retrain_tf


def main():

    
    parser = argparse.ArgumentParser(description='MNIST drawing demo.')
    parser.add_argument("-r","--recompile", action="store_true", help="Whether to recompile the model")
    parser.add_argument("-b","--backend", choices=("torch","tf"), default="torch", help="Neural network library backend")
    parser.add_argument("-v", "--verbose",  action="store_true", default=False, help="Level of verbosity")
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    if args.recompile:
        retrain_pyt()
    else:
        app = MNISTApplication(verbose=args.verbose)
        app.mainloop()
        app.window.close()


if __name__ == "__main__":
    main()