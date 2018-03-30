import os
import subprocess
from utils import print_time_info


def REPEATSEQ(data_dir):
    input_data = []
    output_labels = [[], [], [], []]
    with open(os.path.join(data_dir, "data.txt"), 'r') as file:
        data_size = int(subprocess.getoutput("wc -l {}".format(
            os.path.join(data_dir, "data.txt"))).split(' ')[0])
        for l_idx, line in enumerate(file):
            if l_idx % 1000 == 0:
                print_time_info(
                        "Processed {}/{} lines".format(l_idx, data_size))
            _input, _output = line.strip().split(' | ')
            input_data.append(_input.split(' '))
            _output = _output.split(' ')
            for idx in range(4):
                output_labels[idx].append(_output)

    return input_data, output_labels
