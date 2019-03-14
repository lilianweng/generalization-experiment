import json
import os
import subprocess
import numpy as np
from utils import colorize, configure_logging
import logging

configure_logging()

MAX_LOG_UNITS = 14.0


class NewRiskCurveExperiment:

    def __init__(self, num_epoches):
        self.output_filename = "data/risk_curve_results.json"
        self.num_epoches = num_epoches
        self.start_i, self.results = self._preload_results()

    def _preload_results(self):
        start_i = 0.5

        if os.path.exists(self.output_filename):
            with open(self.output_filename, 'r') as fin:
                results = json.load(fin)
            if len(results) > 0:
                assert results[0]['n_epoches'] == self.num_epoches
                start_i = max(r['i'] for r in results) + 0.5
        else:
            results = []

        logging.info(f"preloaded results: {results}")
        logging.info(f"n_epoches:{self.num_epoches}")
        logging.info(f"start_i:{start_i}")

        return start_i, results

    def run(self):
        for i in np.arange(self.start_i, MAX_LOG_UNITS + 0.5, step=0.5):
            n_units = int(np.exp(i))

            # We are training MNIST models using subproces because tensorflow graph cannot be
            # dynamically removed. In this way, we keep only one graph in memory in each loop.
            proc = subprocess.run(
                ['python', 'risk_curve_evaluate_model.py',
                 '--n-units', str(n_units), '--n-epoches', str(self.num_epoches)],
                encoding='utf-8', stdout=subprocess.PIPE
            )
            output = proc.stdout.strip().split('\n')
            train_loss, train_acc, eval_loss, eval_acc, eval_err = list(
                map(float, output[-1].split()[1:]))
            result_dict = dict(
                i=i,
                n_units=n_units,
                n_epoches=self.num_epoches,
                train_loss=train_loss,
                train_accuracy=train_acc,
                eval_loss=eval_loss,
                eval_accuracy=eval_acc,
                eval_error=eval_err
            )
            self.results.append(result_dict)
            logging.info(colorize(f"n_units={n_units} >>> {result_dict}", 'green'))

            # save to disk in every loop
            with open(self.output_filename, 'w') as fout:
                json.dump(self.results, fout)

        return self.results

    def plot(self):
        assert os.path.exists(self.output_filename)
        with open(self.output_filename, 'r') as fin:
            results = json.load(fin)

        import matplotlib.pyplot as plt
        num_units = [x['n_units'] for x in results];
        print(num_units)
        train_errors = [1.0 - x['train_accuracy'] for x in results]
        errors = [x['eval_error'] for x in results]
        plt.plot(num_units, errors, 'x-')
        plt.plot(num_units, train_errors, '.-')
        plt.xscale('log')
        plt.grid(color='k', ls='--', alpha=0.4)
        plt.show()


if __name__ == '__main__':
    exp = NewRiskCurveExperiment(10)
    exp.run()
