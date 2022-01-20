import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

nn_scores = pd.read_csv('f1_scores/evaluation_nn.csv')[:9]
knn_scores = pd.read_csv('f1_scores/evaluation_knn.csv')[:9]
rf_scores = pd.read_csv('f1_scores/evaluation_rf.csv')[:9]

nn_scores['Classifier'] = 'Neural Network'
knn_scores['Classifier'] = 'KNN'
rf_scores['Classifier'] = 'Random Forest'

df = pd.concat([nn_scores, knn_scores, rf_scores], axis=0, ignore_index=True)
df['Score Type'] = 'F1-Macro'
df_copy = df.copy()
df_copy['F1-Macro'] = df_copy['F1-Micro']
df_copy['Score Type'] = 'F1-Micro'

combined_scores = pd.concat([df, df_copy], ignore_index=True)

combined_scores['F1-Macro'] = combined_scores['F1-Macro'] * 100

# g = sns.catplot(x='Dataset', y='F1-Micro', hue='Classifier', data=df, kind='bar', height=5, palette='dark', aspect=1.5)
#
# (g.set_axis_labels(ylabel='F1-Macro')
#  .set_titles('Dataset')
#  .set_xticklabels(df['Dataset'].drop_duplicates(), rotation=45)
#  .despine(left=True)
#  .set(ylim=(0, 1))
#  .legend.set_title('Classifier')
#  )
#
# plt.subplots_adjust(bottom=0.5)
# plt.show()

sns_plot = sns.relplot(x="F1-Macro", y="Dataset", hue="Classifier", s=100, style="Score Type", data=combined_scores,
                       aspect=2, alpha=0.8)
sns_plot.set_axis_labels('F1-Score', '')
sns_plot.set(title="F1-Macro vs. F1-Micro on different datasets")
plt.subplots_adjust(bottom=0.1, top=0.9)
fig = sns_plot.fig
fig.show()


nn_scores = pd.read_csv('f1_scores/evaluation_nn.csv')[11:]
knn_scores = pd.read_csv('f1_scores/evaluation_knn.csv')[11:]
rf_scores = pd.read_csv('f1_scores/evaluation_rf.csv')[11:]

nn_scores['Classifier'] = 'Neural Network'
knn_scores['Classifier'] = 'KNN'
rf_scores['Classifier'] = 'Random Forest'

df = pd.concat([nn_scores, knn_scores, rf_scores], axis=0, ignore_index=True)
df['F1-Weighted'] = df['F1-Weighted'] * 100
df['Embedding'] = df['method']
df.sort_values(by='Dataset', inplace=True)

sns_plot = sns.relplot(x="F1-Weighted", y="Dataset", hue="Classifier", s=100, style="Embedding", data=df,
                       aspect=2, alpha=0.8)
sns_plot.set_axis_labels('Weighted F1-Score', '')
sns_plot.set(title="Ridle vs. other KG Embeddings")
plt.subplots_adjust(bottom=0.1, top=0.9)
fig = sns_plot.fig
fig.show()
