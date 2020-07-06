# Testing cProfile for filtering module outputs

import cProfile, pstats, io, sys, os, subprocess, argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Profiler arguments.')

    parser.add_argument('--file', type=str,
                        help='file to run.')
    parser.add_argument('--lib', type=str,
                        help='Network library: networkx (n) or snapx (s).')
    parser.set_defaults(file='ncc', lib='n')

    return parser.parse_args()


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
        subprocess.call(['./transform_bench.sh'])

    enablePrint()

    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats("^.*([^d][^e][^e][^p]snap|networkx).*")
    print(s.getvalue())

    with open(f"db/cprof/{choice}_{lib}x.txt", "w+") as f:
        f.write(s.getvalue())

def main():
    args = arg_parse()

    assert(args.file in ['lnc', 'gc', 'lnwn', 'ncc', 'gct'])
    assert(args.lib in ['n', 's'])

    profile_script(args.file, args.lib)

if __name__ == "__main__":
    main()
