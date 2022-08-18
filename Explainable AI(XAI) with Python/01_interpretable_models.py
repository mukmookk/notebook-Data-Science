# %% Imports
from utils import DataLoader
from interpret.glassbox import (LogisticRegression,
                                ClassificationTree,
                                ExplainableBoostingClassifier)
from interpret import show
from sklearn.metrics import f1_score, accuracy_score

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocessing_data()

# %% Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
print(X_train.shape)
print(X_test.shape)
print(y_train.value_counts(0))

# %% Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print("After oversampling: ", X_train.shape)

print(y_train.value_counts(0))

# %% Fit logistic regression model
lr = LogisticRegression(random_state=2022, feature_names=X_train.columns, penalty='l1', solver="liblinear")

lr.fit(X_train, y_train)
print("Training finished")

# %% Evaluate logistic regression model
y_pred = lr.predict(X_test)
print(f"F1 score {f1_score(y_test, y_pred, average='micro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain global logistic regression model
lr_local = lr.explain_local(X_test[:], y_test[:], name='Logistic Regression')
show(lr_local)

# %% Explain global logistic regression model
lr_global = lr.explain_global(name='Logistic Regression')
show(lr_global)

# %% Fit decision tree model
tree = ClassificationTree()
tree.fit(X_train, y_train)
print("Training finished")
y_pred =  tree.predict(X_test)
print(f"F1 score {f1_score(y_test, y_pred, average='micro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain local prediction
tree_local = tree.explain_local(X_test, y_test, name='Tree')
show(tree_local)

# %% Fit Explainable Boosting Machine
ebm = ExplainableBoostingClassifier(random_state=2022)
ebm.fit(X_train, y_train)
print("Training finished.")
y_pred = ebm.predict(X_test)
print(f"F1 score {f1_score(y_test, y_pred, average='micro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain locally
ebm_local = ebm.explain_local(X_test[:100], y_test[:100], name='EBM')
show(ebm_local)

# %% Explain globally
ebm_global = ebm.explain_global(name='EBM')
show(ebm_global)

# %%
