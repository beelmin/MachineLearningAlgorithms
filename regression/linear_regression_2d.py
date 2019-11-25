from typing import List, Tuple
from sklearn import linear_model
import numpy as np
import plotly.offline as ply
import plotly.graph_objs as go


car_mileage_vs_value_data_set: List[List[int]] = [
    (10000, 31000), (400000, 19000), (5000, 32000), (0, 40000),
    (1000, 33000), (100000, 26000), (50000, 29000), (50, 35000),
    (20000, 30000), (200000, 20000)]


car_mileage_vs_value_data_set: np.ndarray = np.array(car_mileage_vs_value_data_set)
car_mileages: np.ndarray = car_mileage_vs_value_data_set.T[0]
car_values: np.ndarray = car_mileage_vs_value_data_set.T[1]

plot_data = go.Scatter(x=car_mileages, 
                       y=car_values, 
                       mode='markers',
                       marker=dict(color='black'))

layout = go.Layout(xaxis=dict(ticks='', showticklabels=True,
                              zeroline=False),
                   yaxis=dict(ticks='', showticklabels=True,
                              zeroline=False),
                   showlegend=False, hovermode='closest')


expanded_car_mileages: np.ndarray = np.expand_dims(car_mileages, axis=1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training set
regr.fit(expanded_car_mileages, car_values)

# The mean squared error
sklearn_mse: float = np.mean((regr.predict(expanded_car_mileages) - car_values) ** 2)
print(f'Mean squared error: {sklearn_mse}')

linear_regression_line = go.Scatter(x=car_mileages, 
                                    y=regr.predict(expanded_car_mileages),
                                    mode='lines',
                                    line=dict(color='blue', width=3))

figure = go.Figure(data=[plot_data, linear_regression_line], layout=layout)

ply.iplot(figure)



# Test
test_mileages: np.ndarray = np.array([150000, 250000, 300000])
expanded_test_mileages: np.ndarray = np.expand_dims(test_mileages, axis=1)
approximated_values: np.ndarray = regr.predict(expanded_test_mileages)

for mileage, value in zip(test_mileages, approximated_values):
    print(f'Mileage: {mileage} -> Value: {round(value)}')









