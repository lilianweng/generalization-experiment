import subprocess
import time
import logging
import json
import os
from utils import configure_logging

os.makedirs('data', exist_ok=True)
configure_logging()


def main(net_layers):
    net_layers = [64]
    net_layers_str = '-'.join(map(str, net_layers))
    output_filename = f"data/intrinsic-dimension-net-{net_layers_str}-{int(time.time())}.json"
    logging.info(f"output_filename: {output_filename}")

    results = {}
    for d in [1, 5]:  # range(1, 1501, 50):
        proc = subprocess.run(
            ['python', 'intrinsic_dimensions_measurement.py',
             ','.join(map(str, net_layers)), str(d)],
            encoding='utf-8', stdout=subprocess.PIPE
        )
        last_line = proc.stdout.strip().split('\n')[-1]
        final_eval_acc = float(last_line)
        results[d] = final_eval_acc

    with open(output_filename, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main([64])
