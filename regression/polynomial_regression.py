from typing import List, Tuple
import plotly.offline as ply
import plotly.graph_objs as go
import numpy as np


def calculate_polynomial_mse(training_set: List[Tuple[float, float]], *coefficients: float) -> float:
    polynomial = np.poly1d(coefficients)
    
    return sum([(polynomial(x) - y)**2 for x, y in training_set]) / len(training_set)


# Instantiate dataset
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
                       marker=dict(color='black'),
                       name='Training point')

layout = go.Layout(xaxis=dict(ticks='', showticklabels=True,
                              zeroline=False),
                   yaxis=dict(ticks='', showticklabels=True,
                              zeroline=False),
                   showlegend=False, hovermode='closest')



m_2, m_1, m_0 = np.polyfit(
    x=car_mileages,
    y=car_values,
    deg=2)


polynomial = np.poly1d([m_2, m_1, m_0])

x_new = np.linspace(min(car_mileages), max(car_mileages), 50)
y_new = polynomial(x_new)

regression_curve = go.Scatter(
    x=x_new,
    y=y_new,
    mode='lines',
    line=dict(color='blue', width=3),
    name='Regression curve')

figure = go.Figure(data=[plot_data,
                         regression_curve],
                   layout=layout)

ply.iplot(figure)




polynomial_regression_mse: float = calculate_polynomial_mse(car_mileage_vs_value_data_set, m_2, m_1, m_0)
print(f'Quadratic regression MSE: {polynomial_regression_mse}')


