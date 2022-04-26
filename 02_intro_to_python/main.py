import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
# import this

data = pd.read_csv('lsd_math_score_data.csv')

time = data[['Time_Delay_in_Minutes']].values

lsd = data[['LSD_ppm']].values

score = data[['Avg_Math_Test_Score']].values

plt.title('Tissue Concentration of LSD vs Average Maths Score', fontsize=18)
plt.ylabel('Average Maths Score (% of Control)', fontsize=12)
plt.xlabel('Tissue Concentration of LSD (ppm)', fontsize=12)
plt.text(x=0.5, y=-18, s='Wagner et al. (1968)')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlim(1, 6.5)
plt.ylim(0, 80)

plt.style.use('dark_background')

plt.plot(lsd, score, color='#e74c4c', lw=6)
plt.show()

regr = LinearRegression()
regr.fit(lsd, score)
print(regr.coef_[0][0])
print(regr.intercept_[0])
print(regr.score(lsd, score))
predicted_score = regr.predict(lsd)



plt.scatter(lsd, score)
plt.title('Tissue Concentration of LSD vs Average Maths Score', fontsize=18)
plt.ylabel('Average Maths Score (% of Control)', fontsize=12)
plt.xlabel('Tissue Concentration of LSD (ppm)', fontsize=12)
plt.text(x=0.5, y=15, s='Wagner et al. (1968)')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.plot(lsd, predicted_score, color='red')

plt.show()