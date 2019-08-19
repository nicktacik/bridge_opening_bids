from data_pipeline import load_clean_data
import numpy as np
from network_model import train_val_test_split, get_network_model
from itertools import product
from data_models import BasicModel, AdvancedModel
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from os.path import isdir
from os import mkdir


def run_experiment(data, params, output_stream):
    patience = 10
    data_model = BasicModel() if params['model'] == 'basic' else AdvancedModel()
    network_size = [1024, 256, 64] if params['network'] == 'small' else [2048, 1024, 512, 256, 128, 64]
    dropout_frac = 0.25 if params['dropout'] == 'low' else 0.5

    train_gen, val_data, test_data = train_val_test_split(data, data_model)
    save_name = params['model'] + '_' + params['activation'] + '_' + params['dropout'] + '_' + params['network']
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    cp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True,
                         filepath='./experiment_results/' + save_name + '.h5', )
    network_model = get_network_model(data_model, params['activation'], network_size, dropout_frac)
    history = network_model.fit_generator(train_gen, epochs=250,
                                          steps_per_epoch=int(0.5 * len(data) / params['batch_size']),
                                          validation_data=val_data, callbacks=[es, cp])

    train_acc = history.history['acc'][-patience - 1]
    train_loss = history.history['loss'][-patience - 1]
    val_acc = history.history['val_acc'][-patience - 1]
    val_loss = history.history['val_loss'][-patience - 1]
    network_model = load_model('./experiment_results/' + save_name + '.h5')
    test_loss, test_acc = network_model.evaluate(test_data[0], test_data[1])
    output_stream.write(
        "model = {0}\n train_loss/acc = {1}\t{2}\n val_loss/acc = {3}\t{4} \n test_loss/acc ={5}\t{6}\n\n".format(
            save_name, round(train_loss, 3), round(train_acc, 3), round(val_loss, 3),
            round(val_acc, 3), round(test_loss, 3), round(test_acc, 3)
        )
    )


def eda():
    data = load_clean_data()
    num_uniuqe_hands = len(data)
    num_total_hands = sum(len(d) for d in data)
    num_pass = sum(1.0 for d in data for h in d if h.bid == 'Pass')
    print("Number of unique hands: {0}".format(num_uniuqe_hands))
    print("Number of total hands: {0}".format(num_total_hands))
    print("Fraction of passes: {:.2f}".format(100.0 * num_pass / num_total_hands))

    pcts = []
    totals = []
    for hands in data:
        if len(hands) >= 2:
            all_bids = set(hand.bid for hand in hands)
            most_bids = max(len([hand for hand in hands if hand.bid == bid]) for bid in all_bids)
            totals.append(len(hands))
            pcts.append(most_bids * 1.0 / len(hands))
    weight_avg = sum(p * t for p, t in zip(pcts, totals)) / sum(totals)
    print("Weighted average of best guesses: {:.2f}".format(100.0 * weight_avg))


def main():
    experiment_params = {
        'model': ['basic', 'advanced'],
        'activation': ['tanh', 'relu'],
        'dropout': ['low', 'high'],
        'network': ['small', 'large'],
        'batch_size': [64]
    }
    data = load_clean_data()
    np.random.seed(42)
    np.random.shuffle(data)

    if not isdir('./experiment_results'):
        mkdir('./experiment_results')

    with open('./experiment_results/results.txt', 'w') as ouput_file:
        for params in (dict(zip(experiment_params, x)) for x in product(*experiment_params.values())):
            run_experiment(data, params, ouput_file)


if __name__ == '__main__':
    main()
