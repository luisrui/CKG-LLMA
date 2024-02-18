from modules.utils import *
import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--argpath', type=str, default='args.yaml', help='the relative path of argments file')
    args = parse.parse_args()
    args = read_yaml(path=args.argpath)
    