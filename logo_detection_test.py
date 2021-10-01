import tensorflow as tf

GRAPH_PB_PATH = '/home/andrehus/egne_prosjekter/videoAndOutput/models/transnetv2-weights/saved_model.pb'
with tf.compat.v1.Session() as sess:
   print("load graph")
   with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.compat.v1.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)