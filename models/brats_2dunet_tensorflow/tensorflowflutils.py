import tensorflow as tf

def tf_get_vars(optimizer=None):
    # FIXME: how to support multiple graphs?
    if optimizer is not None:
        opt_vars = optimizer.variables()
    else:
        opt_vars = []
    return tf.trainable_variables() + opt_vars


def tf_get_tensor_dict(session, vars):
    # FIXME: do this in one call?
    return {var.name: val for var, val in zip(vars, session.run(vars))}


# FIXME: what's a nicer construct than this? ugly interface. Perhaps we get an object with an assumed interface that lets is set/get these?
# Note that this will return the assign_ops and placeholder nodes it uses
# if called with None, it will create them.
# to avoid inflating the graph, caller should keep these and pass them back
# What if we want to set a different group of vars in the middle? It is good if it is the subset of the original vars.
def tf_set_tensor_dict(tensor_dict, session, vars, assign_ops=None, placeholders=None):
    if placeholders is None:
        placeholders = {v.name: tf.placeholder(v.dtype, shape=v.shape) for v in vars}
    if assign_ops is None:
        assign_ops = {v.name: tf.assign(v, placeholders[v.name]) for v in vars}

    for k, v in tensor_dict.items():
        session.run(assign_ops[k], feed_dict={placeholders[k]:v})
    
    return assign_ops, placeholders

def tf_reset_vars(session, vars):
    for var in vars:
        var.initializer.run(session=session)