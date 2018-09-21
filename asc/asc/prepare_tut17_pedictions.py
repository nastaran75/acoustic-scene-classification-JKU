
import glob
import argparse
import subprocess

from train import select_model


def cmd(cmdline):
    try:
        output = subprocess.check_output(cmdline, shell=True)
        print output
    except Exception as e:
        print 'cmdline', cmdline
        print 'e', e
        exit()

if __name__ == "__main__":
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Produce results for model.')
    parser.add_argument('--data', help='tut17 or tut17_test.')
    parser.add_argument('--fold', help='train split (only considered if data is set to eval).', type=str)
    parser.add_argument('--model', help='select model to evaluate.')
    args = parser.parse_args()

    model = select_model(args.model)

    params = glob.glob("/home/matthias/experiments/asc/%s_model_search/params_%s_*.pkl" % (model.EXP_NAME, args.fold))
    for p in params:
        command = 'python eval.py --model %s --data %s --fold %s --params \"%s\" --dump_results .mat' % (args.model, args.data, args.fold, p)
        cmd(command)
