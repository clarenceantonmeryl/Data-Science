import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import valuation_tool as val

import matplotlib.pyplot as plt
import seaborn as sns
import time

# Gather data
boston_dataset = load_boston()
# print(boston_dataset)
# print(dir(boston_dataset))
print(boston_dataset.DESCR)

# Data points and features
#
# print(boston_dataset.data.shape)
# print(boston_dataset.feature_names)
# Prices in 1000s
# print(boston_dataset.target)
#
# Data Exploration with Pandas dataframes

data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)


data['PRICE'] = boston_dataset.target

# Top
print(data.head())

# Bottom
print(data.tail())

print(data.count())

# Cleaning data - Check for missing values

print(pd.isnull(data).any())
print(data.info())

# Data Visualisation - Histogram, Distributions, Bar Charts

'''
plt.figure(figsize=(10, 6))
plt.hist(x=data['PRICE'], bins=50, ec='black', color='#6667AB')
plt.title('House Prices in Boston')
plt.xlabel('Price in 1000s')
plt.ylabel("Number of Houses")
plt.show()

sns.distplot(data['PRICE'], bins=50, hist=True, kde=False, color='#fbc02d')
plt.show()

plt.hist(x=data['RM'], ec='black', color='#fbc02d')
plt.title('Number of Rooms in a Boston House')
plt.xlabel('Number of Rooms')
plt.ylabel('Number of Houses')
plt.show()

sns.distplot(data['RM'], color='#fbc02d')
plt.show()

print(data['RM'].mean())

print(data['RAD'].value_counts())
plt.hist(x=data['RAD'], bins=25, ec='black', color='#6667AB', rwidth=0.75)
plt.title('Accessibility to Radial Highways')
plt.xlabel('Radial Highway Accessibility')
plt.ylabel('Number of Houses')
plt.grid()
plt.show()

frequency = data['RAD'].value_counts()

plt.figure(figsize=(20, 12))
print(frequency.index)
print(frequency.axes[0])
plt.bar(frequency.index, height=frequency)
plt.title('Accessibility to Radial Highways', fontsize=20)
plt.xlabel('Radial Highway Accessibility', fontsize=20)
plt.ylabel('Number of Houses', fontsize=20)
plt.show()

'''

print(data['CHAS'].value_counts())

# Descriptive Statistics

print(f"Price min: {data['PRICE'].min()}")
print(f"Price max: {data['PRICE'].max()}")
print(f"Price mean: {data['PRICE'].median()}")
print(f"Price median: {data['PRICE'].mean()}")

print(f"Data min: {data.min()}")
print(f"Data max: {data.max()}")
print(f"Data mean: {data.median()}")
print(f"Data median: {data.mean()}")

print(data['PRICE'].describe())

"""
# Correlation

print(data['PRICE'].corr(data['RM']))
print(data['PRICE'].corr(data['PTRATIO']))

correlation = data.corr(method='pearson')

print(correlation)

mask = np.zeros_like(correlation)
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

print(mask)

# Plot

plt.figure(figsize=(16, 10))

sns.heatmap(correlation, mask=mask, annot=True, annot_kws={"size": 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.show()

dis_nox_corr = round(data['NOX'].corr(data['DIS']), 3)

plt.scatter(x=data['DIS'], y=data['NOX'], s=50, alpha=0.3, color="indigo")
plt.title(f'DIS vs NOX (Correlation: {dis_nox_corr})', fontsize=14)
plt.xlabel('DIS')
plt.ylabel('NOX')
# plt.show()

# Seaborn

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['DIS'], y=data['NOX'], height=7, color='indigo', joint_kws={'alpha': 0.3})
# plt.show()

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['DIS'], y=data['NOX'], kind='hex', height=7, color='blue')
# plt.show()

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['DIS'], y=data['NOX'], kind='reg', height=7, color='green')
# plt.show()

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['DIS'], y=data['NOX'], kind='resid', height=7, color='red')
# plt.show()

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['DIS'], y=data['NOX'], kind='kde', height=7, color='orange')
# plt.show()

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['TAX'], y=data['RAD'], height=7, color='orange', kind='reg')
# plt.show()

sns.lmplot(x='TAX', y='RAD', data=data, height=7)
# plt.show()

# RM vs PRICE
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['RM'], y=data['PRICE'], height=7, color='lightblue', kind='reg')
# plt.show()

start = time.process_time()
sns.pairplot(data)
plt.show()
end = time.process_time()
print(end - start)


start = time.process_time()
sns.pairplot(data, kind='reg', plot_kws={'line_kws': {'color': 'cyan'}})
plt.show()
end = time.process_time()
print(end - start)


# Training and Dataset Split

prices = data['PRICE']
features = data.drop('PRICE', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    prices,
    test_size=0.2,
    random_state=10
)

print(len(X_train) / len(features))
print(X_test.shape[0] / features.shape[0])

# Multivariable Regression

regr = LinearRegression()
regr.fit(X_train, y_train)

print(f'Intercept: {regr.intercept_}')
print(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']))

print(f'Training r^2: {regr.score(X_train, y_train)}')
print(f'Test r^2: {regr.score(X_test, y_test)}')

# Data Transformations

print(data['PRICE'].skew())

y_log = np.log(data['PRICE'])
print(y_log)

print(y_log.skew())

sns.set()
# sns.distplot(y_log)
# plt.title(f'Log price with skew {y_log.skew()}')
# plt.show()

transformed_data = features
transformed_data['LOG_PRICE'] = y_log

sns.lmplot(
    x='LSTAT',
    y='PRICE',
    data=data,
    height=7,
    scatter_kws={'alpha': 0.3},
    line_kws={'color': 'darkred'}
)

plt.show()

sns.lmplot(
    x='LSTAT',
    y='LOG_PRICE',
    data=transformed_data,
    height=7,
    scatter_kws={'alpha': 0.3},
    line_kws={'color': 'cyan'}
)

plt.show()

"""

# Training and Dataset Split (Log Prices)

prices = np.log(data['PRICE'])
features = data.drop('PRICE', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    prices,
    test_size=0.2,
    random_state=10
)

# Multivariable Regression (Log Prices)

regr = LinearRegression()
regr.fit(X_train, y_train)

print(f'Intercept: {regr.intercept_}')
print(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']))

print(f'Training r^2: {regr.score(X_train, y_train)}')
print(f'Test r^2: {regr.score(X_test, y_test)}')

# Charles River Property Premium

print(np.e**0.080331)

# P-values and Evaluating Coefficients

X_include_constant = sm.add_constant(X_train)

model = sm.OLS(y_train, X_include_constant)
results = model.fit()

# print(results.params)
# print(results.pvalues)

print(pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 3)}))

# Testing for Multicollinearity

# print(variance_inflation_factor(exog=X_include_constant.values, exog_idx=1))

# vif = []
#
# for index in range(len(X_include_constant.columns)):
#     # if index != 0:
#     vif.append(variance_inflation_factor(exog=X_include_constant.values, exog_idx=index))
#

vif = [
    variance_inflation_factor(exog=X_include_constant.values, exog_idx=index)
    for index in range(len(X_include_constant.columns))
]

print(vif)

vif_df = pd.DataFrame(
    {
        'coef_name': X_include_constant.columns,
        'vif': np.around(vif, 2)
    }
             )

print(vif_df)

# Model Simplication and Baysian Information Criterion

# Original Model
X_include_constant = sm.add_constant(X_train)

model = sm.OLS(y_train, X_include_constant)
results = model.fit()


org_coef = pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 3)})


# print(f"BIC: {results.bic}")
# print(f"R-squared: {results.rsquared}")

# Reduced model (Exclude INDUS)

X_include_constant = sm.add_constant(X_train)
X_include_constant = X_include_constant.drop(['INDUS'], axis=1)

model = sm.OLS(y_train, X_include_constant)
results_model_2 = model.fit()

coef_exclude_indus = pd.DataFrame({'coef': results_model_2.params, 'p-value': round(results_model_2.pvalues, 3)})


# print(f"BIC: {results_model_2.bic}")
# print(f"R-squared: {results_model_2.rsquared}")

# Reduced model (Exclude INDUS and AGE)

X_include_constant = sm.add_constant(X_train)
X_include_constant = X_include_constant.drop(['INDUS', 'AGE'], axis=1)

model = sm.OLS(y_train, X_include_constant)
results_model_3 = model.fit()

coef_exclude_indus_age = pd.DataFrame({'coef': results_model_3.params, 'p-value': round(results_model_3.pvalues, 3)})


# print(f"BIC: {results_model_3.bic}")
# print(f"R-squared: {results_model_3.rsquared}")

# Testing Multicollinearity

frames = [org_coef, coef_exclude_indus, coef_exclude_indus_age]
# print(pd.concat(frames, axis=1))

# Residuals and Residual Plots

# Modified model: log prices and exclude INDUS, AGE
prices = np.log(data['PRICE'])
features = data.drop(['PRICE', 'INDUS', 'AGE'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    prices,
    test_size=0.2,
    random_state=10
)

# Using Statsmodel
X_include_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, X_include_constant)
results = model.fit()

# Residuals
# residuals = y_train - results.fittedvalues
# results.resid

# Graph of Actual vs Predicted Prices

plt.figure(figsize=(16, 12))
sns.set()
corr = round(y_train.corr(results.fittedvalues), 2)
print(corr)

plot = sns.jointplot(
    x=y_train,
    y=results.fittedvalues,
    color='navy',
    kind='reg',
    line_kws={'color': 'red'}
)
plot.set_axis_labels('Actual Log Prices $y _i$', 'Predicted Log Prices $\hat y _i$')
plot.figure.tight_layout()
plot.fig.suptitle(f"Actual vs Predicted Log Prices: $y _i$ vs $\hat y _i$ (corr: {corr})", color='red')
plt.show()

plt.scatter(x=y_train, y=results.fittedvalues, c='navy', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.xlabel('Actual Log Prices $y _i$')
plt.ylabel('Predicted Log Prices $\hat y _i$')
plt.title(f"Actual vs Predicted Log Prices: $y _i$ vs $\hat y _i$ (corr: {corr})")
plt.show()

plt.scatter(x=np.e**y_train, y=np.e**results.fittedvalues, c='blue', alpha=0.6)
plt.plot(np.e**y_train, np.e**y_train, color='cyan')
plt.xlabel('Actual Prices (1000s) $y _i$')
plt.ylabel('Predicted Prices (1000s) $\hat y _i$')
plt.title(f"Actual vs Predicted Prices: $y _i$ vs $\hat y _i$ (corr: {corr})")
plt.show()

# Residuals vs Predicted Values
plt.scatter(x=results.fittedvalues, y=results.resid, c='blue', alpha=0.6)

plt.xlabel('Predicted Log Prices $\haty _i$')
plt.ylabel('Residuals')
plt.title(f"Residuals vs Fitted Values")
plt.show()

# Distribution of Residuals (log prices) - Checking for Normality
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color='navy')
plt.title(f'Log Price Model: Residuals, Skew ({resid_skew}), Mean ({resid_mean})')
plt.show()

# MSE and R Squared
reduced_log_mse = round(results.mse_resid, 3)
reduced_log_rsquared = round(results.rsquared, 3)


# Original Model
prices = data['PRICE']
features = data.drop(['PRICE'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    prices,
    test_size=0.2,
    random_state=10
)


X_include_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, X_include_constant)
results = model.fit()

# Actual vs Predicted Prices (Original Model)

corr = round(y_train.corr(results.fittedvalues), 2)
plt.scatter(x=y_train, y=results.fittedvalues, c='orange', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.xlabel('Actual Prices (1000s) $y _i$')
plt.ylabel('Predicted Prices (1000s) $\hat y _i$')
plt.title(f"Actual vs Predicted Prices: $y _i$ vs $\hat y _i$ (corr: {corr})")
plt.show()


# Residuals vs Predicted Values (Original Model)
plt.scatter(x=results.fittedvalues, y=results.resid, c='green', alpha=0.6)

plt.xlabel('Predicted Prices $\haty _i$')
plt.ylabel('Residuals')
plt.title(f"Residuals vs Fitted Values")
plt.show()

# Distribution of Residuals (Original Model)
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color='indigo')
plt.title(f'Price Model: Residuals, Skew ({resid_skew}), Mean ({resid_mean})')
plt.show()
# MSE and R Squared
full_normal_mse = round(results.mse_resid, 3)
full_normal_rsquared = round(results.rsquared, 3)


# Model without Key Features using Log Prices
prices = np.log(data['PRICE'])
features = data.drop(['PRICE', 'INDUS', 'AGE', 'LSTAT', 'RM', 'NOX', 'CRIM'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    prices,
    test_size=0.2,
    random_state=10
)


X_include_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, X_include_constant)
results = model.fit()

# Actual vs Predicted Prices (Full Reduced Model)

corr = round(y_train.corr(results.fittedvalues), 2)
plt.scatter(x=y_train, y=results.fittedvalues, c='#e74c3c', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.xlabel('Actual Prices (Log) $y _i$')
plt.ylabel('Predicted Prices (Log) $\hat y _i$')
plt.title(f"Actual vs Predicted Log Prices with Omitted Variables: $y _i$ vs $\hat y _i$ (corr: {corr})")
plt.show()

# Residuals vs Predicted Values (Full Reduced Model)
plt.scatter(x=results.fittedvalues, y=results.resid, c='#e74c3c', alpha=0.6)

plt.xlabel('Predicted Prices (Log) $\haty _i$')
plt.ylabel('Residuals')
plt.title(f"Residuals vs Fitted Values")
plt.show()

# Distribution of Residuals (Original Model)
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color='#e74c3c')
plt.title(f'Price Model: Residuals, Skew ({resid_skew}), Mean ({resid_mean})')
plt.show()

# MSE and R Squared
omitted_var_mse = round(results.mse_resid, 3)
omitted_var_rsquared = round(results.rsquared, 3)

# MSE and R Squared Analysis
mse_rsquared = pd.DataFrame(
    {
        'MSE': [reduced_log_mse, full_normal_mse, omitted_var_mse],
        'RMSE': np.sqrt([reduced_log_mse, full_normal_mse, omitted_var_mse]),
        'R-Squared': [reduced_log_rsquared, full_normal_rsquared, omitted_var_rsquared],

     },
    index=['Reduced Log Model', 'Full Normal Price Model', 'Omitted Variable Model']
)

print(mse_rsquared)

print(f"2 standard deviations: {2 * mse_rsquared['RMSE'][0]}")
std_2 = 2 * mse_rsquared['RMSE'][0]
print(f"Upper Bound log Prices {np.log(30) + std_2}")
print(f"Lower Bound log Prices {np.log(30) - std_2}")
print(f"Upper Bound Prices 1000s {np.e**(np.log(30) + std_2)}")
print(f"Lower Bound Prices 1000s {np.e**(np.log(30) - std_2)}")

# Use Valuation Tool

print(val.estimate_house_price(nr_rooms=9, ptratio=24, next_to_river=True, high_confidence=True))
