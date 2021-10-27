
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-map')
    parser.add_argument('-p', '--prefix', default='african')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    infile = open(args.input_map)
    all_samples = list(infile)

    random.shuffle(all_samples)

    n=len(all_samples)



    n_train1 = int(n * 720 / 1020)
    n_train2 = int(n * 120 / 1020)
    n_val = n - n_train1 - n_train2

    print('Total: {}   train1: {}   train2: {}   val: {}'.format(n, n_train1, n_train2, n_val))

    train1_file = open(args.prefix + '_train1.map', 'w')
    train2_file = open(args.prefix + '_train2.map', 'w')
    val_file = open(args.prefix + '_val.map', 'w')

    for x in all_samples[:n_train1]:
        train1_file.write(x)
    for x in all_samples[n_train1:n_train1+n_train2]:
        train2_file.write(x)
    for x in all_samples[-n_val:]:
        val_file.write(x)

    train1_file.close()
    train2_file.close()
    val_file.close()