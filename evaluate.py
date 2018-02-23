import sys
import argparse

default_eyepacs_dir = "./data/eyepacs"
default_messidor2_dir = "./data/messidor2"

parser = argparse.ArgumentParser(
                    description="Evaluate performance of trained graph "
                                "on test data set. "
                                "You should run this with either -m or -e.")
parser.add_argument("-m", "--messidor2", action="store_true",
                    help="evaluate performance on Messidor-2")
parser.add_argument("-e", "--eyepacs", action="store_true",
                    help="evaluate performance on EyePacs set")
parser.add_argument("--data_dir", help="directory where data set resides")

args = parser.parse_args()

if bool(args.eyepacs) == bool(args.messidor2):
    print("Specify either -e or -m!")
    parser.print_help()
    sys.exit(2)

if args.data_dir is not None:
    data_dir = str(args.data_dir)
elif args.eyepacs:
    data_dir = default_eyepacs_dir
elif args.messidor2:
    data_dir = default_messidor2_dir
