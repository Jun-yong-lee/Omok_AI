import numpy as np
from glob import glob
from tqdm import tqdm
import os

game_rule = 'Standard'
base_path = 'dataset'

output_path = os.path.join('dataset_1515', os.path.basename(base_path))
os.makedirs(output_path, exist_ok=True)

file_list = glob(os.path.join('dataset_1515', '%s*/*.psq' % (game_rule, )))

for index, file_path in enumerate(tqdm(file_list)):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    w, h = 7, 7

    lines = lines[1:]

    inputs, outputs = [], []
    board = np.zeros([h, w], dtype=np.int8)

    for i, line in enumerate(lines):
        if ',' not in line:
            break
        x, y, t = np.array(line.split(','), np.int8)

        if x >= 8 or y >= 8:
            break

        if i % 2 == 0:
            player = 1
        else:
            player = 2

        input1 = board.copy().astype(np.int8)
        input1[(input1 != player) & (input1 != 0)] = -1
        input1[(input1 == player) & (input1 != 0)] = 1

        output = np.zeros([h, w], dtype=np.int8)
        output[y-1, x-1] = 1

        for k in range(4):
            input_rot = np.rot90(input1, k=k)
            output_rot = np.rot90(output, k=k)

            inputs.append(input_rot)
            outputs.append(output_rot)

            inputs.append(np.fliplr(input_rot))
            outputs.append(np.fliplr(output_rot))

            inputs.append(np.flipud(input_rot))
            outputs.append(np.flipud(output_rot))

        board[y-1, x-1] = player

# dataset 저장
    np.savez_compressed(os.path.join(output_path, '%s.npz' % (str(index).zfill(5))), inputs=inputs, outputs=outputs)