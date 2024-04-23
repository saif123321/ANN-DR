from flask import Flask, redirect, url_for, render_template, request, flash , jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/trained-dataset", methods=["POST"])
def trained_dataset():
    if 'csv_file' in request.files:
        csv_file = request.files['csv_file']
        df = pd.read_csv(csv_file)
        accuracy, conf_matrix, report = train_and_evaluate_model(df)
        response = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix.tolist(),  
            "classification_report": report
        }
        return jsonify(response), 200
    else:
        return jsonify({'error': 'No file provided'}), 400

def train_and_evaluate_model(df):
    df = df.drop(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Dataset'], axis=1)
    df = pd.get_dummies(df)
    X = df.drop(['Label'], axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, conf_matrix, report

if __name__ == "__main__":
    app.run(debug=True)