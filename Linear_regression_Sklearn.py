import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# Load dataset
data = pd.read_csv(
    r"D:\ML Git\Regression\placement_predict_50k.csv"
)

# Select features and target
x = data[
    [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]
]

y = data["Salary Package"]

# Remove missing values
data = pd.concat([x, y], axis=1).dropna()

x = data[
    [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]
]

y = data["Salary Package"]

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Predict salary
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
root_mse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", root_mse)
print("R2 Score:", r2)

# Take input from the user
cgpa = float(input("Enter CGPA: "))
aptitude = float(input("Enter AptitudeTestScore: "))
coding = float(input("Enter CodingTestScore: "))
interview = float(input("Enter InterviewTestScore: "))

# Create DataFrame with same feature names
new_student = pd.DataFrame(
    [[cgpa, aptitude, coding, interview]],
    columns=[
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]
)

# Predict salary
salary = model.predict(new_student)

# Extract the single prediction from NumPy array
salary = salary[0]

print("\nPredicted Salary Package:", round(salary, 2), "LPA")

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("actual salary package")
plt.ylabel("predicted salary package")
plt.title("Predicted Salary Package")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
)
plt.show()

