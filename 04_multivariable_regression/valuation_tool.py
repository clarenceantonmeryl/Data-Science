from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather data
boston_dataset = load_boston()

data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)

features = data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(data=log_prices, columns=['PRICE'])
# print(target.shape)

CRIME = 0
ZN = 1
CHAS = 2
RM = 4
PTRATIO = 8
MEDIAN_BOSTON_PRICE = 583.3 * 1000  # 728730

# property_stats = np.ndarray(shape=(1, 11))
# property_stats[0][CRIME] = features['CRIM'].meam()
# property_stats[0][ZN] = features['ZN'].mean()
# property_stats[0][CHAS] = features['CHAS'].mean()

property_stats = features.mean().values.reshape(1, 11)
print(property_stats)

regr = LinearRegression().fit(features, target)
fitted_values = regr.predict(features)

MSE = mean_squared_error(target, fitted_values)
RMSE = np.sqrt(MSE)


def estimate_house_price(nr_rooms,
                         ptratio,
                         next_to_river=False,
                         high_confidence=True):

    """
    Estimate the price of a property in Boston.

    :param nr_rooms: Number of Rooms in Property
    :param ptratio: Pupil to Teacher Ratio
    :param next_to_river: If the property is next to the Carles River
    :param high_confidence: Range of values
    :return: Estimated house price in Boston with upper and lower bounds
    """

    # Configure Property

    if nr_rooms < 1 or ptratio < 1 or ptratio > 60:
        print('That is unrealistic. Try again')
        return

    property_stats[0][RM] = nr_rooms
    property_stats[0][PTRATIO] = ptratio
    if next_to_river:
        property_stats[0][CHAS] = 1
    else:
        property_stats[0][CHAS] = 0

    # Make Prediction
    log_estimate = regr.predict(property_stats)[0][0]
    # Calc range

    if high_confidence:
        upper_bound = log_estimate + 2 * RMSE
        lower_bound = log_estimate - 2 * RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68

    inflation = MEDIAN_BOSTON_PRICE / (np.median(boston_dataset.target) * 1000)

    price_estimate = np.around(((np.e ** log_estimate) * inflation) * 1000, -3)

    upper_bound = np.around(((np.e ** upper_bound) * inflation) * 1000, -3)

    lower_bound = np.around(((np.e ** lower_bound) * inflation) * 1000, -3)

    print(f"The estimated property price is {price_estimate}")
    print(f"At {interval}% confidence, the calculation range is USD {lower_bound} to {upper_bound}")

    return price_estimate, interval, lower_bound, upper_bound


# estimate_house_price(nr_rooms=9, ptratio=27, next_to_river=True, high_confidence=False)
