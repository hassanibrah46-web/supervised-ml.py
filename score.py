import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#prepare the dataset using python lists
x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y=[30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

#create a linear regression model
model = LinearRegression()

#train the model
model.fit(x, y)

study_hours = float(input("Enter hours studied: "))

#predict examination score 
predicted_score = model.predict([[study_hours]])

#display prediction to two decimal places
print(f"Predicted examination score: {predicted_score[0]}")

#bonus task: plot the graph
#plt.scatter(x, y, color='blue', label='data points')
plt.plot(x, model.predict(x), color='red', label='regression line')
plt.xlabel('Study Hours')
plt.ylabel('Examination Score')
plt.title('Study Hours vs Examination Score')
plt.legend()
plt.grid(True)
plt.show()
