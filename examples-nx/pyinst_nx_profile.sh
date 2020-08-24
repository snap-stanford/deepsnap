# NetworkX pyinst profiling
pyinstrument -o db/pyinst/gc_nx.txt --show-all graph_classification_TU.py
pyinstrument -o db/pyinst/lpc_nx.txt --show-all link_prediction_cora.py
pyinstrument -o db/pyinst/lpwn_nx.txt --show-all link_prediction_wn.py
pyinstrument -o db/pyinst/ncc_nx.txt --show-all node_classification_cora.py
pyinstrument -o db/pyinst/gct_e1_nx.txt --show-all graph_classification_TU_transform.py --transform_batch ego --radius 1
pyinstrument -o db/pyinst/gct_e2_nx.txt --show-all graph_classification_TU_transform.py --transform_batch ego --radius 2
pyinstrument -o db/pyinst/gct_e3_nx.txt --show-all graph_classification_TU_transform.py --transform_batch ego --radius 3
pyinstrument -o db/pyinst/gct_e4_nx.txt --show-all graph_classification_TU_transform.py --transform_batch ego --radius 4
pyinstrument -o db/pyinst/gct_e5_nx.txt --show-all graph_classification_TU_transform.py --transform_batch ego --radius 5
#pyinstrument -o db/pyinst/sgm_nx.txt --show-all subgraph_matching/train_single_proc.py --n_batches 1000
#pyinstrument -o db/pyinst/bnc_nx.txt --show-all bio_application/node_classification_CC.py
#pyinstrument -o db/pyinst/bnc2_nx.txt --show-all bio_application/node_classification_CC2.py
#pyinstrument -o db/pyinst/bnf_nx.txt --show-all bio_application/node_classification_FF.py
