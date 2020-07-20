# Testing cProfile for filtering module outputs

import cProfile, pstats, io, sys, os, subprocess, argparse

# def arg_parse_prof():
#     parser_prof = argparse.ArgumentParser(description='Profiler arguments.')
#
#     parser_prof.add_argument('--file', type=str,
#                         help='file to run.')
#     parser_prof.add_argument('--lib', type=str,
#                         help='Network library: networkx (n) or snapx (s).')
#     parser_prof.set_defaults(file='ncc', lib='n')
#
#     return parser_prof.parse_args()
#

# Disable Print
def blockPrint():
        sys.stdout = open(os.devnull, 'w')

# Restore Print
def enablePrint():
        sys.stdout = sys.__stdout__

# Profile script

def profile_script(choice, lib):

    pr = cProfile.Profile()
    pr.enable()

    blockPrint()

    if choice == "lnc":
        import link_prediction_cora
        link_prediction_cora.main()
    elif choice == "gc":
        import graph_classification_TU
        graph_classification_TU.main()
    elif choice == "lnwn":
        import link_prediction_wn
        link_prediction_wn.main()
    elif choice == "ncc":
        import node_classification_cora
        node_classification_cora.main()
    elif choice == "gct":
        import graph_classification_TU_transform as gct
        gct.main(['--transform_batch', 'ego', '--radius', '1'])
        # gct.main(['--transform_batch', 'ego', '--radius', '2'])
        # gct.main(['--transform_batch', 'ego', '--radius', '3'])
        # gct.main(['--transform_batch', 'ego', '--radius', '4'])
        # gct.main(['--transform_batch', 'ego', '--radius', '5'])
        # gct.main(['--transform_dataset', 'ego', '--radius', '1'])
        # gct.main(['--transform_dataset', 'ego', '--radius', '2'])
        # gct.main(['--transform_dataset', 'ego', '--radius', '3'])
        # gct.main(['--transform_dataset', 'ego', '--radius', '4'])
        # gct.main(['--transform_dataset', 'ego', '--radius', '5'])
        # gct.main(['--transform_batch', 'path'])
        # gct.main(['--transform_dataset', 'path'])
    elif choice == 'bnc':
        import bio_application
        import bio_application.node_classification_CC
    elif choice == 'bnc2':
        import bio_application
        import bio_application.node_classification_CC2
    elif choice == 'bncf':
        import bio_application.node_classification_FF
    elif choice == 'sgm':
        import subgraph_matching.train as sgm
        sgm.main()
    enablePrint()

    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats("^.*([^d][^e][^e][^p]snap|networkx).*", 20)
    print(s.getvalue())

    with open(f"db/cprof/{choice}_{lib}x.txt", "w+") as f:
        f.write(s.getvalue())

def run_all(lib):
    for choice in ['lnc', 'gc', 'lnwn', 'ncc', 'gct',
                   'bnc', 'bnc2', 'bncf', 'sgm']:
        print(f"Processing {choice}...")
        if choice not in ['sgm', 'bnc', 'bnc2', 'bncf', 'gct']:
            profile_script(choice, lib)

def main():
    # file = 'lnwn'
    # lib = 'n'

    # assert(file in ['lnc', 'gc', 'lnwn', 'ncc', 'gct', 'bnc', 'bnc2', 'blpm', 'bncf'])
    # assert(lib in ['n', 's'])

    run_all(lib='s')

if __name__ == "__main__":
    main()
