"""Command line interface of mnist drawer."""

import argparse

from ._gui_main import MNISTApplication

def main():
    parser = argparse.ArgumentParser(description='MNIST drawing demo.')
    parser.add_argument("-v", "--verbose",  action="store_true", default=False, help="Level of verbosity")
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    app = MNISTApplication(verbose=args.verbose)
    app.mainloop()
    app.window.close()


if __name__ == "__main__":
    main()