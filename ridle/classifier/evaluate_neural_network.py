import argparse

from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

from ridle.classifier import load_embeddings, evaluate_keras_classifier, save_f1_scores


# GELU Activation function
def gelu(x):
    return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


def evaluate_neural_network(dataset):
    print(f'Training Neural Network on dataset {dataset}:\n')

    embeddings, types = load_embeddings(dataset)

    mlb = MultiLabelBinarizer()
    targets = mlb.fit_transform(types)

    print('Training Neural Network...')
    model = Sequential()
    model.add(Dense(embeddings.shape[1], input_dim=embeddings.shape[1]))
    model.add(Activation(gelu, name='Gelu'))
    model.add(Dense(targets.shape[1], activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    f1_scores = evaluate_keras_classifier(model, embeddings, types)

    save_f1_scores(f1_scores, dataset, 'nn')


def main():
    parser = argparse.ArgumentParser(description='Neural Network Prediction using Ridle')
    parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
    parser = parser.parse_args()

    dataset = parser.dataset

    evaluate_neural_network(dataset)


if __name__ == '__main__':
    main()
