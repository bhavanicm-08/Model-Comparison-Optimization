import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\n===== Decision Tree =====")
print("Accuracy:", dt_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, dt_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n===== Random Forest =====")
print("Accuracy:", rf_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


svm_model = SVC(kernel='linear')

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("\n===== Support Vector Machine (SVM) =====")
print("Accuracy:", svm_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, svm_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))


print("\n===== SVM Optimization =====")

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5
)

grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_

optimized_pred = best_svm.predict(X_test)

optimized_accuracy = accuracy_score(y_test, optimized_pred)

print("Best Parameters:", grid_search.best_params_)
print("Optimized SVM Accuracy:", optimized_accuracy)


models = ['Decision Tree', 'Random Forest', 'SVM', 'Optimized SVM']
accuracies = [
    dt_accuracy,
    rf_accuracy,
    svm_accuracy,
    optimized_accuracy
]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.ylim(0.8, 1.0)

for i, value in enumerate(accuracies):
    plt.text(i, value + 0.005, f"{value:.2f}", ha='center')

plt.show()

plt.figure(figsize=(12, 8))

plot_tree(
    dt_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree Visualization")

plt.show()