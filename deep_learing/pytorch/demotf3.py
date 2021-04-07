import tensorflow as tf
import keras.backend as K
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16


def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            # model = tf.keras.models.load_model(model_h5_path)
            # model = MobileNet(alpha=.75, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 32, 32, 3)))
            model = VGG16(input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224, 224, 3)))
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops


print(get_flops(""))
# run_meta = tf.compat.v1.RunMetadata()
# with tf.compat.v1.Session(graph=tf.Graph()) as sess:
#     K.set_session(sess)
#     net = MobileNet(alpha=.75, input_tensor=tf.placeholder('float32', shape=(1,32,32,3)))
#
#     opts = tf.profiler.ProfileOptionBuilder.float_operation()
#     flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
#
#     opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
#     params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
#
#     print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))