# Testing cProfile for filtering module outputs

import cProfile, pstats, io, sys, os

# Disable
def blockPrint():
        sys.stdout = open(os.devnull, 'w')

# Restore
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

    enablePrint()

    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats("^.*([^d][^e][^e][^p]snap|networkx).*")
    print(s.getvalue())

    with open(f"db/cprof/{choice}_{lib}.txt", "w+") as f:
        f.write(s.getvalue())

profile_script("ncc", "sx")
