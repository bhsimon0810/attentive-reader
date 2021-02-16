import tensorflow as tf


class Reader(object):
    def __init__(self, cell_type, hid_size, emb_size, vocab_size, num_labels, pretrained_embs, l2_reg_lambda=0.0):
        self.hid_size = hid_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.cell_type = cell_type

        # placeholders
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="x1")
        self.x1_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="x1_lengths")
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name="x2")
        self.x2_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="x2_lengths")
        self.mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name="mask")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

        # embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            self.embeddings = tf.get_variable(
                name="embeddings",
                shape=[self.vocab_size, self.emb_size],
                initializer=tf.constant_initializer(pretrained_embs),
                dtype=tf.float32
            )
            self.x1_embedded = tf.nn.embedding_lookup(self.embeddings, self.x1)
            self.x2_embedded = tf.nn.embedding_lookup(self.embeddings, self.x2)

        # document encoder
        with tf.variable_scope("x1_encoder"):
            cell_fw = self._get_cell(self.hid_size, self.cell_type)
            cell_bw = self._get_cell(self.hid_size, self.cell_type)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.x1_embedded,
                sequence_length=self.x1_lengths,
                dtype=tf.float32
            )
            self.x1_outputs = tf.concat(outputs, axis=-1)

        # query encoder
        with tf.variable_scope("x2_encoder"):
            cell_fw = self._get_cell(self.hid_size, self.cell_type)
            cell_bw = self._get_cell(self.hid_size, self.cell_type)
            _, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.x2_embedded,
                sequence_length=self.x2_lengths,
                dtype=tf.float32
            )
            self.x2_outputs = tf.concat(output_states, axis=-1)

        # bilinear layer
        with tf.variable_scope("bilinear"):
            W_bilinear = tf.Variable(
                tf.random_uniform((2 * self.hid_size, 2 * self.hid_size), minval=-0.01, maxval=0.01))
            M = self.x1_outputs * tf.expand_dims(tf.matmul(self.x2_outputs, W_bilinear), axis=1)
            alphas = tf.nn.softmax(tf.reduce_sum(M, axis=2))
            self.outputs = tf.reduce_sum(self.x1_outputs * tf.expand_dims(alphas, axis=2), axis=1)

        # softmax layer
        with tf.variable_scope("output"):
            prob = tf.layers.dense(self.outputs, units=self.num_labels, activation=tf.nn.softmax,
                                   kernel_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
            masked_prob = prob * self.mask
            modified_prob = masked_prob / tf.expand_dims(tf.reduce_sum(masked_prob, axis=1), axis=1)
            self.modified_prob = tf.clip_by_value(modified_prob, 1e-7, 1.0 - 1e-7)
            self.predictions = tf.cast(tf.argmax(masked_prob, axis=-1), tf.int32, name="predictions")

        # accuracy and loss
        with tf.variable_scope("metric"):
            y_onehot = tf.one_hot(self.y, depth=self.num_labels)
            l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2_reg_lambda
            self.loss = tf.reduce_mean(
                -tf.reduce_sum(y_onehot * tf.log(self.modified_prob), reduction_indices=[1])) + l2_reg
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.y), tf.int32))

    @staticmethod
    def _get_cell(hid_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hid_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hid_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hid_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None
