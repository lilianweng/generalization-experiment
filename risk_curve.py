import json
import logging
import subprocess
import time

from utils import colorize, configure_logging

configure_logging()


class NewRiskCurveExperiment:

    def __init__(self, loss_type, max_epochs, n_train_sample):
        self.output_filename = f"data/risk_curve_loss-{loss_type}_sample-{n_train_sample}_" \
            f"epoch-{max_epochs}_{int(time.time())}.json"
        logging.info(f"output_filename: {self.output_filename}")

        self.loss_type = loss_type
        self.n_train_sample = n_train_sample
        self.max_epochs = max_epochs
        self.results = []

        critical_n_units = int((n_train_sample * 10 - 10) / float(28 * 28 + 10))
        logging.info(f"critical_n_units: {critical_n_units}")
        self.n_units_to_test = sorted(set(
            list(range(critical_n_units - 8, critical_n_units + 4)) +
            list(range(5, 105, 5))
        ))
        logging.info(f"n_units_to_test: {self.n_units_to_test}")

    def run(self):
        for i in range(len(self.n_units_to_test)):
            n_units = self.n_units_to_test[i]
            old_n_units = None if i == 0 else self.n_units_to_test[i - 1]

            # We are training MNIST models using subprocess because tensorflow graph cannot be
            # dynamically removed. In this way, we keep only one graph in memory in each loop.
            args = [
                '--n-units', str(n_units),
                '--max-epochs', str(self.max_epochs),
                '--n-train-samples', str(self.n_train_sample),
                '--loss-type', str(self.loss_type),
            ]

            if old_n_units:
                args.extend(['--old-n-units', str(old_n_units)])

            proc = subprocess.run(
                ['python', 'risk_curve_evaluate_model.py'] + args,
                encoding='utf-8', stdout=subprocess.PIPE
            )
            output = proc.stdout.strip().split('\n')
            epoch, step, train_loss, train_acc, eval_loss, eval_acc = \
                list(map(float, output[-1].split()[1:]))
            result_dict = dict(
                n_epochs=int(epoch),
                step=int(step),
                n_units=n_units,
                old_n_units=old_n_units,
                total_params=(28 * 28 + 1) * n_units + (n_units + 1) * 10,
                train_loss=train_loss,
                train_acc=train_acc,
                eval_loss=eval_loss,
                eval_acc=eval_acc,
            )
            self.results.append(result_dict)
            logging.info(colorize(f"n_units={n_units} >>> {result_dict}", 'green'))

            # save to disk in every loop
            with open(self.output_filename, 'w') as fout:
                json.dump(self.results, fout)

        return self.results

    def plot(self):
        pass


if __name__ == '__main__':
    exp = NewRiskCurveExperiment(loss_type='mse', max_epochs=1000, n_train_sample=2500)
    exp.run()
