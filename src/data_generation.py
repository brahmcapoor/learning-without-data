import numpy as np 
import os
import sys
import random
import json

from typing import List


MAX_ORDER = 3

DATA_PATH = "../data/synthetic_data"

def generate_inputs(num_inputs, min_val, max_val):
	return np.array([random.uniform(min_val, max_val) for _ in range(num_inputs)])

def get_coefficients():
	return [random.uniform(-1, 1) for _ in range(MAX_ORDER + 1)]

def target_fn(x, coefficients: List[int]):
	result = np.zeros_like(x)
	for i, coeff in enumerate(coefficients[::-1]):
		result += coeff * (x ** i)
	return result

def generate_targets(inputs, target_fn):
	std = 1.0
	noise = np.random.normal(scale = std, size = inputs.shape)
	coefficients = get_coefficients()
	true_targets = target_fn(inputs, coefficients)
	targets = true_targets + noise
	return targets, coefficients

def make_info_json(out_path, coefficients):
	json_path = os.path.join(out_path, "info.json")
	json.dump({'coefficients': coefficients}, open(json_path, 'w'))


if __name__ == "__main__":
	num_inputs = 10000
	min_val = -50000
	max_val = 50000
	inputs = generate_inputs(num_inputs, min_val, max_val)
	targets, coefficients = generate_targets(inputs, target_fn)

	out_path = os.path.join(DATA_PATH, sys.argv[1])
	os.makedirs(out_path, exist_ok = True)

	inputs.dump(os.path.join(out_path, "inputs.npz"))
	targets.dump(os.path.join(out_path, "targets.npz"))
	make_info_json(out_path, coefficients)



