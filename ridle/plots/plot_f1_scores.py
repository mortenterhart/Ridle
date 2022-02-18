import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ridle import ROOT_DIR
from ridle.datasets import fb_yago_subsets


def load_f1_scores():
    nn_scores = pd.read_csv(f'{ROOT_DIR}/f1_scores/evaluation_nn.csv')
    knn_scores = pd.read_csv(f'{ROOT_DIR}/f1_scores/evaluation_knn.csv')
    rf_scores = pd.read_csv(f'{ROOT_DIR}/f1_scores/evaluation_rf.csv')

    nn_scores['Classifier'] = 'Neural Network'
    knn_scores['Classifier'] = 'KNN'
    rf_scores['Classifier'] = 'Random Forest'

    return pd.concat([nn_scores, knn_scores, rf_scores], ignore_index=True)


def plot_f1_macro_vs_micro():
    clf_scores = load_f1_scores()

    datasets = ['Books_DBpedia', 'ChemicalCompounds_DBpedia', 'Company_DBpedia', 'dblp', 'DBp_2016-04',
                'Person_DBpedia', 'Songs_DBpedia', 'umls', 'Universities_DBpedia']
    clf_scores = clf_scores[clf_scores['Dataset'].isin(datasets)]

    clf_scores = clf_scores.melt(id_vars=['Classifier', 'Dataset'], value_vars=['F1-Macro', 'F1-Micro'],
                                 var_name='Score Type', value_name='F1-Score')

    clf_scores['F1-Score'] = clf_scores['F1-Score'] * 100

    g = sns.relplot(data=clf_scores, x='F1-Score', y='Dataset', hue='Classifier', s=100, style='Score Type',
                    aspect=2, alpha=0.8)
    g.set_axis_labels('F1-Score', '')
    g.set(title='F1-Macro vs. F1-Micro on different datasets')
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.show()


def plot_ridle_vs_other_embeddings():
    clf_scores = load_f1_scores()

    datasets = fb_yago_subsets.keys()
    clf_scores = clf_scores[clf_scores['Dataset'].isin(datasets)]

    clf_scores.rename(columns={'method': 'Embedding'}, inplace=True)
    clf_scores.sort_values(by='Dataset', inplace=True)

    clf_scores['F1-Weighted'] = clf_scores['F1-Weighted'] * 100

    g = sns.relplot(data=clf_scores, x='F1-Weighted', y='Dataset', hue='Classifier', s=100, style='Embedding',
                    aspect=2, alpha=0.8)
    g.set_axis_labels('Weighted F1-Score', '')
    g.set(title='Ridle vs. other KG Embeddings')
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.show()


def main():
    sns.set_theme(style='whitegrid')

    plot_f1_macro_vs_micro()
    plot_ridle_vs_other_embeddings()


if __name__ == '__main__':
    main()
