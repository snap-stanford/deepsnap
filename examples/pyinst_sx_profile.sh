# NetworkX pyinst profiling
pyinstrument -o db/pyinst/gc_sx.txt --show-all graph_classification_TU.py
pyinstrument -o db/pyinst/lpc_sx.txt --show-all link_prediction_cora.py
pyinstrument -o db/pyinst/lpwn_sx.txt --show-all link_prediction_wn.py
pyinstrument -o db/pyinst/gct_sx.txt --show-all graph_classification_TU_transform.py
pyinstrument -o db/pyinst/bnc_sx.txt --show-all bio_application/node_classification_CC.py
pyinstrument -o db/pyinst/bnc2_sx.txt --show-all bio_application/node_classification_CC2.py
pyinstrument -o db/pyinst/bnf_sx.txt --show-all bio_application/node_classification_FF.py
