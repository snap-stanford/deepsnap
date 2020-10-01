import os
import argparse
import subprocess

# Examples can be runned by:
#	pyinstrument -o db/pyinst/gct.txt --show-all graph_classification_TU.py
#	pyinstrument -o db/pyinst/lpc.txt --show-all link_prediction_cora.py
#	pyinstrument -o db/pyinst/lpwn.txt --show-all link_prediction_wn.py
#	pyinstrument -o db/pyinst/ncc.txt --show-all node_classification_cora.py
#	pyinstrument -o db/pyinst/gctt_ego_1.txt --show-all graph_classification_TU_transform.py --transform_batch ego --radius 1
#	pyinstrument -o db/pyinst/sgm_sp_1000.txt --show-all subgraph_matching/train_single_proc.py --n_batches 1000
#	pyinstrument -o db/pyinst/bio_cf.txt --show-all bio_application/node_classification_FF.py

if os.path.isdir("db") == False:
	process = subprocess.run(["mkdir", "db"])

if os.path.isdir("db/pyinst") == False:
	process = subprocess.run(["mkdir", "db/pyinst"])

def arg_parse():
	parser = argparse.ArgumentParser(description='Pyinstrument profile arguments.')

	parser.add_argument('--file', type=str,
						help='The example file to run.')
	parser.add_argument('--device', type=str,
						help='CPU / GPU device.')
	parser.add_argument('--transform_batch', type=str,
						help='apply transform to each batch.')
	parser.add_argument('--radius', type=int,
						help='Radius of mini-batch ego networks')
	parser.add_argument('--n_batches', type=int,
						help='Number of training minibatches')
	parser.add_argument('--num_graphs', type=int,
						help='Number of graphs in link_prediction_cora.py')

	parser.set_defaults(
			file='node_classification_cora.py',
			device='cuda',
			transform_batch=None,
			radius=0,
			n_batches=0,
			num_graphs=1,
	)
	return parser.parse_args()

def main():
	args = arg_parse()
	fname = ""
	for elem in args.file.split("_"):
		fname += elem[0].lower()
	if "subgraph_matching" in args.file:
		fname = "sgm_" + fname
	if "bio_application" in args.file:
		fname = "bio_" + fname
	fname = "db/pyinst/" + fname
	if "gctt" in fname:
		if args.radius == 0:
			raise ValueError("Please specify the radius/transform_batch to profile graph_classification_TU_transform.")
		fname += "_{}_{}.txt".format(args.transform_batch, args.radius)
		process = subprocess.run(["pyinstrument", "-o", fname, "--show-all", args.file, "--transform_batch", args.transform_batch, "--radius", str(args.radius), "--device", args.device])
	elif "tsp" in fname:
		if args.n_batches == 0:
			raise ValueError("Please specify the n_batches to profile subgraph_matching/train_single_proc.")
		fname += "_{}.txt".format(args.n_batches)
		process = subprocess.run(["pyinstrument", "-o", fname, "--show-all", args.file, "--n_batches", str(args.n_batches), "--device", args.device])
	elif "lpc" in fname:
		fname += "_{}.txt".format(args.num_graphs)
		cmd = ["pyinstrument", "-o", fname, "--show-all", args.file, "--device", args.device, "--num_graphs", str(args.num_graphs)]
		if args.num_graphs >= 10:
			cmd.append("--multigraph")
		process = subprocess.run(cmd)
	else:
		fname += ".txt"
		process = subprocess.run(["pyinstrument", "-o", fname, "--show-all", args.file, "--device", args.device])
	

if __name__ == "__main__":
	main()