import numpy as np
import pickle
from collections import Counter


def load_data(in_file, max_example=None, relabeling=False):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """

    documents = []
    questions = []
    answers = []
    num_examples = 0
    f = open(in_file, 'r', encoding='UTF-8')
    while True:
        line = f.readline()
        if not line:
            break
        question = line.strip().lower()
        answer = f.readline().strip()
        document = f.readline().strip().lower()

        if relabeling:
            q_words = question.split(' ')
            d_words = document.split(' ')
            assert answer in d_words

            entity_to_idx = {}
            entity_id = 0
            for word in d_words + q_words:
                if (word.startswith('@entity')) and (word not in entity_to_idx):
                    entity_to_idx[word] = '@entity' + str(entity_id)
                    entity_id += 1

            q_words = [entity_to_idx[w] if w in entity_to_idx else w for w in q_words]
            d_words = [entity_to_idx[w] if w in entity_to_idx else w for w in d_words]
            answer = entity_to_idx[answer]

            question = ' '.join(q_words)
            document = ' '.join(d_words)

        questions.append(question)
        answers.append(answer)
        documents.append(document)
        num_examples += 1

        f.readline()
        if (max_example is not None) and (num_examples >= max_example):
            break
    f.close()
    print('Examples: %d' % len(documents))
    return (documents, questions, answers)


def build_dict(examples, filepath, max_words=50000):
    """
        Build a dictionary for the words in `sentences` and a dictionary for the entities.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    documents, questions, answers = examples
    word_count = Counter()
    for sent in documents + questions:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    print('Words: %d -> %d' % (len(word_count), len(ls)))

    # leave 0 to PAD
    # leave 1 to UNK
    word_dict = {w[0]: idx + 2 for (idx, w) in enumerate(ls)}
    word_dict['PAD'] = 0
    word_dict['UNK'] = 1
    entity_markers = ['<unk_entity>'] + list(set([w for w in word_dict.keys() if w.startswith('@entity')] + answers))
    entity_dict = {w: idx for (idx, w) in enumerate(entity_markers)}

    # save the dicts
    with open('{}-word-dict.pkl'.format(filepath), 'wb') as f:
        pickle.dump(word_dict, f)
    with open('{}-entity-dict.pkl'.format(filepath), 'wb') as f:
        pickle.dump(entity_dict, f)

    return word_dict, entity_dict


def vectorize(examples, word_dict, entity_dict,
              filepath, sort_by_len=True):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_mask: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_mask = np.zeros((len(examples[0]), len(entity_dict))).astype(float)
    in_y = []
    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
        d_words = d.split(' ')
        q_words = q.split(' ')
        assert (a in d_words)
        seq1 = [word_dict[w] if w in word_dict else 1 for w in d_words] # 1 for unknown word
        seq2 = [word_dict[w] if w in word_dict else 1 for w in q_words] # 1 for unknown word
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1.append(seq1)
            in_x2.append(seq2)
            in_mask[idx, [entity_dict[w] for w in d_words if w in entity_dict]] = 1.0
            in_y.append(entity_dict[a] if a in entity_dict else 0) # 0 for unknown entity


    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_mask = in_mask[sorted_index]
        in_y = [in_y[i] for i in sorted_index]

    with open('{}.pkl'.format(filepath), 'wb') as f:
        pickle.dump((in_x1, in_x2, in_mask, in_y), f)


if __name__ == '__main__':
    train_examples = load_data('data/cnn/train.txt', max_example=100000)
    word_dict, entity_dict = build_dict(train_examples, 'data/cnn')
    vectorize(train_examples, word_dict, entity_dict, 'data/cnn-train')

    dev_examples = load_data('data/cnn/dev.txt', max_example=500)
    vectorize(dev_examples, word_dict, entity_dict, 'data/cnn-dev')

    test_examples = load_data('data/cnn/test.txt', max_example=500)
    vectorize(test_examples, word_dict, entity_dict, 'data/cnn-test')