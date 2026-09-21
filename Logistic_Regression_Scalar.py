import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def load_data(filename):
    data = pd.read_csv(r"D:\ML Git\Regression\placement_predict_50k.csv")

    features = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    x = data[features]
    y = data["PlacementStatus"]

    # Remove missing values
    data = pd.concat([x, y], axis=1).dropna()

    x = data[features]
    y = data["PlacementStatus"]

    return x, y


def split_data(x, y):
    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_and_evaluate(
    x_train,
    y_train,
    x_test,
    y_test,
    name
):
    # Create Logistic Regression model
    model = LogisticRegression(max_iter=500)

    # Train the model
    model.fit(x_train, y_train)

    # Predictions
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    # Accuracy
    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    print(name, "Train Accuracy:", round(train_accuracy, 4))
    print(name, "Test Accuracy:", round(test_accuracy, 4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_pred)

    print(name, "Confusion Matrix:")
    print(cm)

    return model


def main():

    print("--------------")
    print("Logistic Regression")
    print("--------------")

    # CSV file
    filename = "placement.csv"

    # Load data
    x, y = load_data(filename)

    # Split data
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Train and evaluate
    model = train_and_evaluate(
        x_train,
        y_train,
        x_test,
        y_test,
        "Unscaled"
    )


if __name__ == "__main__":
    main()