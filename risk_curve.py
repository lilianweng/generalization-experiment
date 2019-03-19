import json
import os
import subprocess
import numpy as np
from utils import colorize, configure_logging
import logging

configure_logging()

MAX_LOG_UNITS = 14.0


class NewRiskCurveExperiment:
    output_filename = "data/risk_curve_results.json"

    def __init__(self, num_epoches):
        self.num_epoches = num_epoches
        # self.start_n_units, self.results = self._preload_results()
        self.start_n_units, self.results = 0, []

    # def _preload_results(self):
    #     start_n_units = 0
    #
    #     if os.path.exists(self.output_filename):
    #         with open(self.output_filename, 'r') as fin:
    #             results = json.load(fin)
    #         if len(results) > 0:
    #             assert results[0]['n_epoches'] == self.num_epoches
    #             start_i = max(r['i'] for r in results) + 0.5
    #     else:
    #         results = []
    #
    #     logging.info(f"preloaded results: {results}")
    #     logging.info(f"n_epoches:{self.num_epoches}")
    #
    #     return start_n_units, results

    def run(self):
        for n_units in [1, 5, 10, 20, 30, 40] + list(range(50, 101, 5)):
            if n_units <= self.start_n_units:
                continue

            # We are training MNIST models using subproces because tensorflow graph cannot be
            # dynamically removed. In this way, we keep only one graph in memory in each loop.
            proc = subprocess.run(
                ['python', 'risk_curve_evaluate_model.py', '--n-units', str(n_units)],
                encoding='utf-8', stdout=subprocess.PIPE
            )
            output = proc.stdout.strip().split('\n')
            print(">>>> output:", output)
            train_loss, train_acc, train_mse, eval_loss, eval_acc, eval_mse = list(
                map(float, output[-1].split()[1:]))
            result_dict = dict(
                n_units=n_units,
                total_params=(28 * 28 + 1) * n_units + (n_units + 1) * 10,
                n_epoches=self.num_epoches,
                train_loss=train_loss,
                train_accuracy=train_acc,
                train_mse=train_mse,
                eval_loss=eval_loss,
                eval_accuracy=eval_acc,
                eval_mse=eval_mse,
                eval_error=1.0 - eval_acc,
            )
            self.results.append(result_dict)
            logging.info(colorize(f"n_units={n_units} >>> {result_dict}", 'green'))

            # save to disk in every loop
            with open(self.output_filename, 'w') as fout:
                json.dump(self.results, fout)

        return self.results

    @classmethod
    def plot(cls):
        assert os.path.exists(cls.output_filename)
        with open(cls.output_filename, 'r') as fin:
            results = json.load(fin)

        import matplotlib.pyplot as plt
        num_units = [x['total_params'] for x in results];
        print(num_units)
        train_errors = [1.0 - x['train_accuracy'] for x in results]
        errors = [x['eval_error'] for x in results]
        plt.plot(num_units, errors, 'x-')
        plt.plot(num_units, train_errors, '.-')
        plt.xscale('log')
        plt.grid(color='k', ls='--', alpha=0.4)
        plt.show()


if __name__ == '__main__':
    exp = NewRiskCurveExperiment(num_epoches=1)
    exp.run()
