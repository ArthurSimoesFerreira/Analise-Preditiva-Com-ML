import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from pandas.plotting import table
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Carregando o Dataset
df = pd.read_csv("./dataset/games.csv")
print(df.head())

# Pré-processamento -----------------------------------------------------------------------------
# Verificar valores ausentes
print(df.isnull().sum())

# Remover linhas com valores ausentes
df = df.dropna()

# Remover linhas duplicadas
df = df.drop_duplicates()

# Garantir que os tipos de dados estão corretos
print(df.dtypes)

# Verificar a distribuição das classes
print(df["HOME_TEAM_WINS"].value_counts())

# Seleção de Features e Target
features = [
    "FG_PCT_home",
    "FT_PCT_home",
    "FG3_PCT_home",
    "AST_home",
    "REB_home",
    "FG_PCT_away",
    "FT_PCT_away",
    "FG3_PCT_away",
    "AST_away",
    "REB_away",
]
target = "HOME_TEAM_WINS"

X = df[features]
y = df[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Aplicando UnderSampling somente no conjunto de treinamento
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print("Distribuição das classes antes do undersampling:")
print(y_train.value_counts())
print("\nDistribuição das classes após o undersampling:")
print(y_train_res.value_counts())

X_train, y_train = X_train_res, y_train_res

# Treinar Logistic Regression com Cross-Validation ----------------------------------------------
lr = LogisticRegression()
lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=5)
lr_mean = np.mean(lr_cv_scores)
lr_std = np.std(lr_cv_scores)
print(f"Logistic Regression Cross-Validation Scores: {lr_cv_scores}")
print(f"Logistic Regression Cross-Validation Mean Score: {lr_mean}")
print(f"Logistic Regression Cross-Validation Std Dev: {lr_std}")

lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]

# Treinar Random Forest com Cross-Validation ----------------------------------------------------
rf = RandomForestClassifier()
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
rf_mean = np.mean(rf_cv_scores)
rf_std = np.std(rf_cv_scores)
print(f"Random Forest Cross-Validation Scores: {rf_cv_scores}")
print(f"Random Forest Cross-Validation Mean Score: {rf_mean}")
print(f"Random Forest Cross-Validation Std Dev: {rf_std}")

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# Os valores de Cross Validation de Regressão Logística e Random Forest. ------------------------
# Med representa o valor médio e DP representa o Desvio Padrão.
results = {
    "Model": ["RL", "RF"],
    "val1": [lr_cv_scores[0], rf_cv_scores[0]],
    "val2": [lr_cv_scores[1], rf_cv_scores[1]],
    "val3": [lr_cv_scores[2], rf_cv_scores[2]],
    "val4": [lr_cv_scores[3], rf_cv_scores[3]],
    "val5": [lr_cv_scores[4], rf_cv_scores[4]],
    "Med": [lr_mean, rf_mean],
    "DP": [lr_std, rf_std],
}

results_df = pd.DataFrame(results).round(7)

# Visualizar a tabela em formato simples com tamanho ajustado das células
fig, ax = plt.subplots(figsize=(12, 3))  # ajustar o tamanho conforme necessário
ax.axis("tight")
ax.axis("off")
tbl = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.5, 1.5)  # Ajustar o tamanho das células
plt.show()

# Avaliar a performance dos modelos -------------------------------------------------------------
# Calcular os scores para Logistic Regression
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_pred)

# Calcular os scores para Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred)

# Criar a tabela de resultados
results = {
    "Model": ["LR", "RF"],
    "Accuracy": [lr_accuracy, rf_accuracy],
    "Precision": [lr_precision, rf_precision],
    "Recall": [lr_recall, rf_recall],
    "F1 Score": [lr_f1, rf_f1],
    "AUC": [lr_auc, rf_auc],
}

results_df = pd.DataFrame(results).round(4)

# Visualizar a tabela em formato simples com tamanho ajustado das células
fig, ax = plt.subplots(figsize=(12, 3))  # ajustar o tamanho conforme necessário
ax.axis("tight")
ax.axis("off")
tbl = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.5, 1.5)  # Ajustar o tamanho das células
plt.show()


# Matriz de confusão de cada modelo -------------------------------------------------------------
# Função para plotar a matriz de confusão
def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Loss", "Win"],
        yticklabels=["Loss", "Win"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_test, lr_pred, 'Matriz de Confusao - Regressão Logística')
plot_confusion_matrix(y_test, rf_pred, 'Matriz de Confusao - Random Forest')

# Importância de Cada Feature no Random Forest --------------------------------------------------
# Importância das características no Random Forest
feature_importances = rf.feature_importances_
features_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

# Visualizar a importância das características
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Importância de Features no Random Forest')
plt.show()

# Curva ROC -------------------------------------------------------------------------------------
# Calcular as curvas ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plotar as curvas ROC
plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()