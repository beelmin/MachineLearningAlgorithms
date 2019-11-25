import numpy as np
import plotly.offline as ply
import plotly.graph_objs as go
from sklearn import linear_model


training_data: np.array = np.array(
    [(10000 ,5000, 31000), (400000, 100000, 19000), (5000, 2500, 32000),
     (0, 0, 40000), (1000, 500, 33000), (100000, 50000, 26000),
     (50000, 25000, 29000), (50, 24, 35000), (20000, 10000, 30000),
     (200000, 100000, 20000)])


car_mileages: np.ndarray = training_data.T[0]
car_ages: np.ndarray = training_data.T[1]
car_values: np.ndarray = training_data.T[2]

plot_data = go.Scatter3d(x=car_mileages, 
                         y=car_ages,
                         z=car_values,
                         mode='markers',
                         marker=dict(
                            size=10,
                            color=car_values,
                            colorscale='Viridis',
                            opacity=0.8))

layout = go.Layout(
    scene = dict(xaxis=dict(title='Mileage',
                            titlefont=dict(color='rgb(200, 200, 200)')),
                 yaxis=dict(title='Age',
                            titlefont=dict(color='rgb(200, 200, 200)')),
                 zaxis=dict(title='Price',
                            titlefont=dict(color='rgb(200, 200, 200)'))))



# Form the training matrix
car_mileages_and_ages: np.ndarray = np.array(list(zip(car_mileages, car_ages)))

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training set
regr.fit(car_mileages_and_ages, car_values)

# The mean squared error
sklearn_mse: float = np.mean((regr.predict(car_mileages_and_ages) - car_values) ** 2)
print(f'Mean squared error: {sklearn_mse}')

# Form plane points
plane: np.array = np.array([regr.predict([[car_mileage, car_age]
                                            for car_mileage in car_mileages])
                              for car_age in car_ages])

sklearn_regression_plane = go.Surface(
    x=car_mileages,
    y=car_ages,
    z=plane)

figure = go.Figure(data=[plot_data, sklearn_regression_plane], layout=layout)
ply.iplot(figure)


# Test
test_data: np.ndarray = np.array([(150000, 100000), (250000, 200000), (300000, 200000)])
approximated_values: np.ndarray = regr.predict(test_data)

for (mileage, age), value in zip(test_data, approximated_values):
    print(f'Mileage: {mileage} km and Age: {round(age / 24 / 365)} years -> Value: {round(value)} $')