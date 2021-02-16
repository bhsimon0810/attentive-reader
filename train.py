import os
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
from model import Reader
from dataset import Dataset
from utils import gen_embeddings, load_dict

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_file", "data/cnn-train.pkl", "Data source for the training data.")
tf.flags.DEFINE_string("dev_file", "data/cnn-dev.pkl", "Data source for the validating data.")
tf.flags.DEFINE_string("word_dict_file", "data/cnn-word-dict.pkl", "Data source for the word dict.")
tf.flags.DEFINE_string("entity_dict_file", "data/cnn-entity-dict.pkl", "Data source for the entity dict.")

# Model Hyperparameters
tf.flags.DEFINE_string("cell_type", "gru", "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: gru)")
tf.flags.DEFINE_integer("emb_size", 50, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hid_size", 50, "Dimensionality of rnn cell units (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_float("max_grad_norm", 5.0,
                      "Maximum value of the global norm of the gradients for clipping (default: 5.0)")
# tf.flags.DEFINE_boolean("debug", True, "Whether it is debug mode i.e. use only first 100 examples")
# tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
# tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
# tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def train():
    print('-' * 50)
    print('Load data files..')
    print('*' * 10 + ' Train')
    train_examples = Dataset(FLAGS.train_file)
    print('*' * 10 + ' Dev')
    dev_examples = Dataset(FLAGS.dev_file)

    print('-' * 50)
    print('Build dictionary..')
    word_dict, entity_dict = load_dict(FLAGS.word_dict_file, FLAGS.entity_dict_file)

    print('-' * 50)
    # Load embedding file
    embeddings = gen_embeddings(word_dict, FLAGS.emb_size, "data/glove.6B.{}d.txt".format(FLAGS.emb_size))

    print('-' * 50)
    print('Creating TF computation graph...')

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            reader = Reader(
                cell_type=FLAGS.cell_type,
                hid_size=FLAGS.hid_size,
                emb_size=FLAGS.emb_size,
                vocab_size=len(word_dict),
                num_labels=len(entity_dict),
                pretrained_embs=embeddings,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(reader.loss, global_step=global_step)
            acc, acc_op = tf.metrics.accuracy(labels=reader.y, predictions=reader.predictions, name="metrics/acc")
            metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
            metrics_init_op = tf.variables_initializer(var_list=metrics_vars)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", reader.loss)
            acc_summary = tf.summary.scalar("accuracy", reader.accuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # dev summaries
            dev_step = 0
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # initialize all variables
            best_dev_acc = 0.0
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for epoch in range(FLAGS.num_epochs):
                print('-' * 50)
                print('{}> epoch: {}'.format(datetime.datetime.now().isoformat(), epoch))
                # print('Start training...')
                for batch in train_examples.batch_iter(FLAGS.batch_size, desc="Training", shuffle=True):
                    mb_x1, mb_x1_lengths, mb_x2, mb_x2_lengths, mb_mask, mb_y = batch
                    feed_dict = {
                        reader.x1: mb_x1,
                        reader.x1_lengths: mb_x1_lengths,
                        reader.x2: mb_x2,
                        reader.x2_lengths: mb_x2_lengths,
                        reader.mask: mb_mask,
                        reader.y: mb_y,
                        reader.is_training: True,
                        reader.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy, _ = sess.run(
                        [train_op, global_step, train_summary_op, reader.loss, reader.accuracy, acc_op], feed_dict
                    )
                    train_summary_writer.add_summary(summaries, step)
                print("training accuracy = {:.2f}".format(sess.run(acc) * 100))

                sess.run(metrics_init_op)
                # Validating process
                for batch in dev_examples.batch_iter(FLAGS.batch_size, desc="Validating", shuffle=False):
                    dev_step += 1
                    mb_x1, mb_x1_lengths, mb_x2, mb_x2_lengths, mb_mask, mb_y = batch
                    feed_dict = {
                        reader.x1: mb_x1,
                        reader.x1_lengths: mb_x1_lengths,
                        reader.x2: mb_x2,
                        reader.x2_lengths: mb_x2_lengths,
                        reader.mask: mb_mask,
                        reader.y: mb_y,
                        reader.is_training: False,
                        reader.dropout_keep_prob: 0.0
                    }
                    summaries, loss, accuracy, _ = sess.run(
                        [dev_summary_op, reader.loss, reader.accuracy, acc_op], feed_dict
                    )
                    dev_summary_writer.add_summary(summaries, global_step=dev_step)
                dev_acc = sess.run(acc) * 100
                print("validating accuracy = {:.2f}".format(dev_acc))

                # model checkpoint
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    path = saver.save(sess, checkpoint_prefix)
                    print("saved model checkpoint to {}".format(path))

                print("current best validating accuracy = {:.2f}".format(best_dev_acc))

            print("{} optimization finished!".format(datetime.datetime.now()))
            print("best validating accuracy = {:.2f}".format(best_dev_acc))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()