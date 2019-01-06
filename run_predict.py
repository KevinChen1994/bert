# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import os

import numpy as np
import tensorflow as tf

import modeling
import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class ParagraphProcess(DataProcessor):
    """自己实现的读取段落训练文件的Processor"""

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.csv')
        with open(file_path, 'r', encoding='UTF-8') as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split('\t')
            label = split_line[0]
            question = split_line[1]
            examples.append(InputExample(guid=guid, text_a=question, label=label))

        return examples

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'dev.csv')
        with open(file_path, 'r', encoding='UTF-8') as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split('\t')
            label = split_line[0]
            question = split_line[1]
            examples.append(InputExample(guid=guid, text_a=question, label=label))

        return examples

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.csv')
        with open(file_path, 'r', encoding='UTF-8') as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split('\t')
            label = split_line[0]
            question = split_line[1]
            examples.append(InputExample(guid=guid, text_a=question, label=label))

        return examples

    def get_labels(self):
        return ['0', '1', '2', '3']


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`.
    将句子A或者句子A与句子B转成每个词的ID，并添加上[CLS][SEP]标签
    input_ids为每个词的ID，句子长度小于max_length，后边补O
    input_mask每个词对应1，其余的为0，表示为哪个是真正的词，哪个不是词是填充的0
    segment_ids句子ID"""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


processor = ParagraphProcess()

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

label_list = processor.get_labels()

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

index2label = {i: label_list[i] for i in range(len(label_list))}

batch_size = 1
num_labels = len(label_list)
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
global graph
input_ids, input_mask, label_ids, segment_ids = None, None, None, None
graph = tf.get_default_graph()
with graph.as_default():
    input_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name=input_ids)
    input_mask_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name=input_mask)
    label_ids_p = tf.placeholder(tf.int32, [batch_size], name=label_ids)
    segment_ids_p = tf.placeholder(tf.int32, [FLAGS.max_seq_length], name=segment_ids)
    total_loss, pre_example_loss, logtis, probabilities = create_model(bert_config=bert_config,
                                                                       is_training=False,
                                                                       input_ids=input_ids_p,
                                                                       input_mask=input_mask_p,
                                                                       segment_ids=segment_ids_p,
                                                                       labels=label_ids_p,
                                                                       num_labels=num_labels,
                                                                       use_one_hot_embeddings=False)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.init_checkpoint))


# 预测方法
def predict(content):
    label = '0'
    example = InputExample(guid=0, text_a=content, label=label)
    feature = convert_single_example(0, example, label_list, FLAGS.max_seq_length, tokenizer)
    input_ids = np.reshape([feature.input_ids], (1, FLAGS.max_seq_length))
    input_mask = np.reshape([feature.input_mask], (1, FLAGS.max_seq_length))
    segment_ids = np.reshape([feature.segment_ids], (FLAGS.max_seq_length))
    label_ids = [feature.label_id]
    global graph

    with graph.as_default():
        feed_dic = {input_ids_p: input_ids, input_mask_p: input_mask, segment_ids_p: segment_ids,
                    label_ids_p: label_ids}
        possibility = sess.run([probabilities], feed_dic)
        possibility = possibility[0][0]
        label_index = np.argmax(possibility)
        label_predict = index2label[label_index]
    data = {'label': label_predict}
    data = json.dumps(data)
    return data


if __name__ == "__main__":
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("max_seq_length")
    content = '0\t本院认为，原、被告虽经人介绍相识，但双方系自愿登记结婚，具有一定的婚前感情基础。双方婚后生育了一个女孩，建立起了一定的夫妻感情。夫妻在共同生活中发生一些矛盾在所难免，只要双方相互关心、相互尊重，珍惜夫妻感情中好的一面，并真正顾念子女的健康成长，仍有和好的希望。原被告的夫妻感情并未确已破裂，原告要求离婚不符合法定离婚条件，依法不予准许。据此，依照《中华人民共和国婚姻法》第三十二条之规定，判决如下：'
    result = predict(content)
    print(result)
