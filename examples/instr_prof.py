from pyinstrument import Profiler
from pyinstrument.renderers import ConsoleRenderer

profiler = Profiler()
profiler.start()

import node_classification_cora
node_classification_cora.main()

session = profiler.stop()
profile_renderer = ConsoleRenderer(unicode=True, color=True, show_all=True, hide='*/torch_geometric/*')

print(profile_renderer.render(session))
