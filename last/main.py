import re
import tensorflow as tf
import jieba
from tensorflow import keras

import os

train_y = []
datas = []
for i in range(7):
    files = os.listdir("./data/{}".format(i+1))
    for file in files:
        with open("./data/{}/".format(i+1)+file, 'r', encoding='utf-8') as f:
            train_y.append(i + 1)
            datas.append(f.read())
train_y = tf.one_hot(train_y, 7)


word_all = set()
for text in datas:
    text = re.sub("[\r|\n|\\s!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·]+", "", text)

    # 结巴分词进行分词
    cut = jieba.lcut(text)
    for word in cut:
        word_all.add(word)

word2id = dict()
for i, word in enumerate(word_all):
    word2id[word] = i

train_x = []
for text in datas:
    # 用正则表达式去掉标点
    text = re.sub("[\r|\n|\\s!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·]+", "", text)

    # 结巴分词进行分词
    data = list()
    cut = jieba.lcut(text)
    for word in cut:
        index = word2id[word]
        data.append(index)
    train_x.append(data)

# 这里我们统计一下每个句子的长度，并形成一个列表
num_tokens = [len(tokens) for tokens in train_x]

# plt.hist(num_tokens, bins=50)
# plt.ylabel('number of tokens')
# plt.xlabel('length of tokens')
# plt.title('Distribution of tokens length')
# plt.show()

is_train = 0

if is_train:
    train_x = keras.preprocessing.sequence.pad_sequences(
        train_x,
        value=0,
        padding='post',  # pre表示在句子前面填充，post表示在句子末尾填充
        maxlen=20
    )

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).shuffle(6500).batch(32)

    example_input_batch, example_target_batch = next(iter(train_ds))
    print(example_target_batch)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(word_all), 64, input_length=train_x.shape[1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2)),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.summary()

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    history = model.fit(train_ds, epochs=15)

    model.save('net_model.h5')
else:
    model = tf.keras.models.load_model("net_model.h5")


def predict_sentiment(_text):
    _text = re.sub("[\r|\n|\\s!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·]+", "", _text)

    _cut = jieba.lcut(_text)
    cut_list = []
    for WORD in _cut:
        _index = word2id[WORD]
        cut_list.append(_index)

    # padding
    tokens_pad = keras.preprocessing.sequence.pad_sequences(
        [cut_list],
        value=0,
        padding='post',  # pre表示在句子前面填充，post表示在句子末尾填充
        maxlen=85)

    # 预测
    output = model.predict(tokens_pad)
    print(output)
    res = output[0].argmax() + 1
    print(res)
    return res


predict_sentiment("云南昭通连续遭遇两起气象灾害共造成6死9伤2011年06月20日22:54中国新闻网中新网昆明6月20日电("
                  "赵书勇)记者20日从云南省昭通市应急办获悉，受受强对流气候影响，昭通市境内连续遭遇两起气象灾害，共造成6人死亡，9人受伤，1人失踪。　　据介绍，受强对流气候影响，6月19日16：20"
                  "左右，云南省昭通市昭阳区小龙洞乡小米村发生单点冰雹袭击，形成洪涝灾害，造成5人死亡，1人失踪，2人重伤，2人轻伤。　　同日下午17时10分至18时30"
                  "分，昭通市彝良县树林乡树林村塘沟、下厂两个村民小组6户6人遭受雷击灾害，造成塘沟村民小组1名村民死亡，下厂村民小组村民5人受伤，其中2人为重伤，其余3"
                  "人为轻伤。灾害发生后，当地政府立即组织相关救援部门赶赴灾区展开救援工作，第一时间将伤者送往医院，排除次生灾害，安抚死者家属。目前，伤者已全部脱离生命危险，救灾工作仍在继续，灾区民众生产生活已恢复正常，相关灾情正在进一步统计当中。　　昭通市地处云贵高原，位于云南省东北部，与贵州、四川两省接壤，地势南高北低，最低海拔267米，最高海拔4040米。境内大山林立，峡谷遍布，是云南气象灾害及地质灾害的重灾区之一。(完)")
