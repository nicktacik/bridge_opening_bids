from random import choice
import numpy as np
from keras import models, layers


def data_generator(data, model, batch_size=64, random_choice=True):
    # a generator that will yield (x, y) in batches
    # if random_choice is True it will choose a hand result randomly
    # otherwise it will choose the first element

    i = 0
    while True:
        out_x = []
        out_y = []
        for j in range(i, batch_size + i):
            if j >= len(data):
                break

            hands = data[j]
            if random_choice:
                hand = choice(hands)
            else:
                hand = hands[0]

            out_x.append(model.encode_hand(hand))
            out_y.append(model.encode_bid(hand.bid))

        i = j + 1
        if i > len(data):
            i = 0

        yield (np.array(out_x), np.array(out_y))


def get_network_model(data_model, activation, layer_sizes, dropout_frac):
    input_shape = (data_model.size(), )
    model = models.Sequential()
    model.add(layers.Dense(layer_sizes[0], activation=activation, input_shape=input_shape))
    model.add(layers.Dropout(dropout_frac))
    for layer_size in layer_sizes:
        model.add(layers.Dense(layer_size, activation=activation))
        model.add(layers.Dropout(dropout_frac))
    model.add(layers.Dense(36, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_val_test_split(data, data_model, batch_size=64, train_frac=0.5, val_frac=0.25):
    train_max_i = int(len(data) * train_frac)
    val_max_i = int(len(data) * val_frac + train_max_i)

    train_gen = data_generator(data[:train_max_i], data_model, batch_size)
    val_data = (
        np.array([data_model.encode_hand(hand[0]) for hand in data[train_max_i:val_max_i]]),
        np.array([data_model.encode_bid(hand[0].bid) for hand in data[train_max_i:val_max_i]])
    )
    test_data = (
        np.array([data_model.encode_hand(hand[0]) for hand in data[val_max_i:]]),
        np.array([data_model.encode_bid(hand[0].bid) for hand in data[val_max_i:]])
    )

    return train_gen, val_data, test_data


