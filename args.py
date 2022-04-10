import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.1,
                        required=False)

    parser.add_argument('--wd',
                        help='weight decay',
                        type=float,
                        default=5e-5,
                        required=False)

    parser.add_argument('--epochs',
                        help='epochs',
                        type=int,
                        default=600,
                        required=False)

    parser.add_argument('--alpha',
                        help='alpha',
                        type=float,
                        default=0.1,
                        required=False)

    parser.add_argument('--beta',
                        help='beta',
                        type=float,
                        default=0.01,
                        required=False)

    parser.add_argument('--gamma',
                        help='gamma',
                        type=float,
                        default=0.1,
                        required=False)

    parser.add_argument('--device',
                        help='cuda:0 or cuda:1',
                        type=int,
                        default=0,
                        required=False)

    return parser.parse_args()
