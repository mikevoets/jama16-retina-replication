import sys
import argparse

default_eyepacs_dir = "./data/eyepacs"
default_messidor2_dir = "./data/messidor2"

parser = argparse.ArgumentParser(
                    description="Evaluate performance of trained graph "
                                "on test data set. "
                                "Specify --data_dir if you use the -o param.")
parser.add_argument("-m", "--messidor2", action="store_true",
                    help="evaluate performance on Messidor-2")
parser.add_argument("-e", "--eyepacs", action="store_true",
                    help="evaluate performance on EyePacs set")
parser.add_argument("-o", "--other", action="store_true",
                    help="evaluate performance on your own dataset")
parser.add_argument("--data_dir", help="directory where data set resides")

args = parser.parse_args()

if bool(args.eyepacs) == bool(args.messidor2) == bool(args.other):
    print("Can only evaluate one data set at once!")
    parser.print_help()
    sys.exit(2)

if args.data_dir is not None:
    data_dir = str(args.data_dir)
elif args.eyepacs:
    data_dir = default_eyepacs_dir
elif args.messidor2:
    data_dir = default_messidor2_dir
elif args.other and args.data_dir is None:
    print("Please specify --data_dir.")
    parser.print_help()
    sys.exit(2)
