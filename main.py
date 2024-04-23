from flask import Flask, redirect, url_for, render_template, request, flash, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

IMAGE_FOLDER = 'static/images'


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/trained-dataset", methods=["POST"])
def trained_dataset():
    required_columns = [
        "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
        "PROTOCOL", "L7_PROTO", "IN_BYTES", "OUT_BYTES", "IN_PKTS",
        "OUT_PKTS", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "Label",
        "Attack", "Dataset"
    ]

    if 'csv_file' in request.files:
        csv_file = request.files['csv_file']
        df = pd.read_csv(csv_file)

        # Check if all required columns are present
        if all(col in df.columns for col in required_columns):
            accuracy, conf_matrix, report, fpr, tpr, roc_auc, svm_model, X_test, y_test, X = train_and_evaluate_model(
                df)

            # Plot 1: Feature Importance
            feature_importance = pd.Series(svm_model.coef_[0], index=X.columns)
            feature_importance.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Relative Importance')
            plt.ylabel('Features')
            plt.savefig(os.path.join(IMAGE_FOLDER, 'feature_importance.png'))
            plt.close()

            # Plot 2: Confusion Matrix
            predictions = svm_model.predict(X_test)
            cm = confusion_matrix(y_test, predictions,
                                  labels=svm_model.classes_)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=svm_model.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(IMAGE_FOLDER, 'confusion_matrix.png'))
            plt.close()

            # Plot 3: ROC Curve
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(IMAGE_FOLDER, 'roc_curve.png'))
            plt.close()

            # Plot 4: Thread Name Classification
            # Assuming 'Actual_Thread_Name' is a column containing actual thread names from your dataset
            actual_thread_names = df['Attack'].values
            predictions = predictions[:len(actual_thread_names)]
            plt.figure(figsize=(10, 8))
            plt.scatter(range(len(actual_thread_names)), actual_thread_names, cmap='viridis')
            plt.colorbar(label='Predicted Thread Name')
            plt.xlabel('Data Points')
            plt.ylabel('Thread Names')
            plt.title('Thread Name Classification')
            plt.savefig(os.path.join(IMAGE_FOLDER, 'thread_classification.png'))
            plt.close()

            response = {
                "accuracy": accuracy,
                # "confusion_matrix": conf_matrix.tolist(),
                "classification_report": report,
                # "plots": ['feature_importance.png', 'confusion_matrix.png', 'roc_curve.png', 'thread_classification.png']
            }
            return jsonify(response), 200
        else:
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400
    else:
        return jsonify({'error': 'No file provided'}), 400

def train_and_evaluate_model(df):
    df = df.drop(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Dataset'], axis=1)
    df = pd.get_dummies(df)
    X = df.drop(['Label'], axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='linear', random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate ROC curve
    y_score = svm_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    return accuracy, conf_matrix, report, fpr, tpr, roc_auc, svm_model, X_test, y_test, X


if __name__ == "__main__":
    app.run(debug=True)
