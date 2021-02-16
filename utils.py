import numpy as np
import pickle


def gen_embeddings(word_dict, dim, in_file=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.uniform(-0.5, 0.5, (len(word_dict), dim)) / dim
    if in_file is not None:
        print('loading pre-trained word embeddings ...')
        embedding_weights = {}
        f = open(in_file, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_weights[word] = vector
        f.close()
        print('total {} word vectors in {}'.format(len(embedding_weights), in_file))

        oov_count = 0
        for word, i in word_dict.items():
            embedding_vector = embedding_weights.get(word)
        if embedding_vector is not None:
            embeddings[i] = embedding_vector
        else:
            oov_count += 1
        print('number of OOV words = %d' % oov_count)

    return embeddings


def save_params(file_name, params, **kwargs):
    """
        Save params to file_name.
        params: a list of Theano variables
    """
    dic = {'params': [x.get_value() for x in params]}
    dic.update(kwargs)
    with open(file_name, "wb") as save_file:
        pickle.dump(dic, save_file)


def load_params(file_name):
    """
        Load params from file_name.
    """
    with open(file_name, "rb") as save_file:
        dic = pickle.load(save_file)
    return dic


def load_dict(word_dict_file_name, entity_dict_file_name):
    print('loading word dict and entity dict ...')

    with open(word_dict_file_name, 'rb') as f:
        word_dict = pickle.load(f)
        print('word dict size = %d' % len(word_dict))

    with open(entity_dict_file_name, 'rb') as f:
        entity_dict = pickle.load(f)
        print('entity dict size = %d' % len(entity_dict))

    return word_dict, entity_dict