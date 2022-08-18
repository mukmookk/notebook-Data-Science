# %% Imports
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from interpret import show
from utils import DataLoader

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocessing_data()

print("Loading and Preprocessing Finished.")

# %% Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
print("Data split at ratio of 8:2, using random state 2022")
print(X_train.shape)
X_train, y_train = data_loader.oversample(X_train, y_train)
print("Oversampling Finished.")
print(X_train.shape)

# %% Fit blackbox model
start_time = time.time()
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
end_time = time.time()

print("Using Random Forest Classifier")
print("Fitting Job is finished.")
print(f"Total fitting time: {end_time - start_time :.2f} second(s)")

y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='micro')}")
print(f"Accuracy score {accuracy_score(y_test, y_pred)}")
# %% Apply lime
# Initilize Lime for Tabular data
lime = LimeTabular(predict_fn=rf.predict_proba,
                   data=X_train,
                   random_state=2022)

# Get Local explaination
lime_local = lime.explain_local(X_test[-20:],
                                y_test[-20:],
                                name='LIME')
show(lime_local)

# %%
