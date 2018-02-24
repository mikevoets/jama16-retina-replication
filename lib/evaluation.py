import tensorflow as tf


def _get_operations_by_names(graph, names):
    return [graph.get_operation_by_name(name) for name in names]


def _get_tensors_by_names(graph, names):
    return [graph.get_tensor_by_name(name) for name in names]


def perform_test(sess, init_op, summary_writer=None, epoch=None,
                 feed_dict_fn=None):
    tf.keras.backend.set_learning_phase(False)
    sess.run(init_op)

    # Retrieve all tensors.
    graph = tf.get_default_graph()
    reset_tp, reset_fp, reset_fn, reset_tn, reset_brier, reset_auc = \
        _get_operations_by_names(
            graph, ['tp/reset', 'fp/reset', 'fn/reset', 'tn/reset',
                    'brier/reset', 'auc/reset'])

    update_tp, update_fp, update_fn, update_tn, update_brier, update_auc, \
    brier, auc, confusion_matrix, summaries_op = \
        _get_tensors_by_names(
            graph, ['tp/op:0', 'fp/op:0', 'fn/op:0', 'tn/op:0',
                    'brier/mean_squared_error/update_op:0',
                    'auc/auc/update_op:0',
                    'brier/mean_squared_error/value:0',
                    'auc/auc/value:0',
                    'confusion_matrix/confusion_matrix:0',
                    'Merge/MergeSummary:0'])

    # Reset all streaming variables.
    sess.run([reset_tp, reset_fp, reset_fn, reset_tn, reset_brier, reset_auc])

    try:
        while True:
            if feed_dict_fn is not None:
                feed_dict = feed_dict_fn()
            else:
                feed_dict = None

            # Retrieve the validation set confusion metrics.
            sess.run([update_tp, update_fp, update_fn,
                      update_tn, update_brier, update_auc], feed_dict)

    except tf.errors.OutOfRangeError:
        pass

    # Retrieve confusion matrix and estimated roc auc score.
    test_conf_matrix, test_brier, test_auc, summaries = sess.run(
        [confusion_matrix, brier, auc, summaries_op])

    # Write summary.
    if summary_writer is not None:
        summary_writer.add_summary(summaries, epoch)

    # Print total roc auc score for validation.
    print(f"Brier score: {test_brier:6.4}, AUC: {test_auc:10.8}")

    # Print confusion matrix.
    print(f"Confusion matrix:")
    print(test_conf_matrix[0])

    return test_auc
