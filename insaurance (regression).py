Path = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from itertools import groupby
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from urllib.request import urlretrieve
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.feature_selection import RFE , RFECV , SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
urlretrieve(Path, 'medical.csv')
import pandas as pd
medical_data = pd.read_csv('medical.csv')
print(medical_data.info())
print(medical_data.describe())
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
int_data = medical_data.select_dtypes(include=['int64', 'float64'])
qual_data = medical_data.select_dtypes(exclude=['int64', 'float64'])

for col in int_data.columns:
    col_distribution = sns.histplot(int_data[col], kde=True)
    col_distribution.set_title(f'Distribution of {col}')

  
    

  
for col in int_data.columns:
    skewness = int_data[col].skew()
    if skewness > 1 or skewness <= -1:
        print(f'{col} is highly skewed with a skewness of {skewness:.2f}')
    elif skewness > 0.5 or skewness < -0.5:
        print(f'{col} is moderately skewed with a skewness of {skewness:.2f}')
    else:
        print(f'{col} is approximately symmetric with a skewness of {skewness:.2f}')  

for col in qual_data.columns:
    col_distribution = sns.countplot(x=qual_data[col])
    col_distribution.set_title(f'Distribution of {col}')
    
for children_count in int_data['children'].unique():
    
    children_distribution = len(int_data[int_data['children'] == children_count]) / len(int_data)
    print(f'Proportion of count->{children_count}: {children_distribution}')

for col in qual_data.columns:
    value_proportion = qual_data[col].value_counts(normalize=True)
    print(f'Value counts for {col}:\n{value_proportion}\n')

target_variable = medical_data['charges']
predictor_variables = medical_data.drop(columns=['charges'])

\

fig_smoker_charges = px.histogram(medical_data, 
                   x='charges', 
                   marginal='box', 
                   color='smoker', 
                   color_discrete_sequence=['green', 'grey'], 
                   title='Annual Medical Charges')
fig_smoker_charges.update_layout(bargap=0.1)
fig_smoker_charges .show()


fig_region_charges = px.histogram(medical_data, 
                   x='charges', 
                   marginal='box', 
                   color='region', 
                   color_discrete_sequence=['blue', 'orange', 'green', 'red'], 
                   title='Annual Medical Charges')
fig_region_charges.update_layout(bargap=0.1)
#fig_region_charges .show()
medical_data['age_bins'] = pd.cut(medical_data['age'], bins=[18, 30, 40, 50, 60, 70], labels=['18-29', '30-39', '40-49', '50-59', '60+'] , include_lowest=False )
medical_data['age_bins'] = medical_data['age_bins'].astype(str)
print(medical_data['age_bins'].value_counts())
fig_age_charges = px.histogram(medical_data, 
                   x='charges', 
                   marginal='box', 
                   color='age_bins', 
                   color_discrete_map={
        '18-29': '#636EFA',
        '30-39': "#EFDD3B",
        '40-49': "#00CC14",
        '50-59': "#ED0000",
        '60+': "#040404"
    },
                   title='Annual Medical Charges')
fig_age_charges.update_layout(bargap=0.1)

fig_age_charges .show()

fig = px.scatter(medical_data, 
                 x='age', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='Age vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


fig = px.scatter(medical_data, 
                 x='bmi', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()
children_avg=medical_data.groupby('children')['charges'].mean().reset_index()
fig_children = px.bar(children_avg, 
                   x='children',
                   y='charges',
                   color='children', 
                   color_discrete_sequence=px.colors.qualitative.Plotly,
                   title='Annual Medical Charges')
fig_children.update_layout(bargap=0.1)
fig_children.show()
children_sum=medical_data.groupby('children')['charges'].sum().reset_index()
fig_children = px.bar(children_sum, 
                   x='children',
                   y='charges',
                   color='children', 
                   color_discrete_sequence=px.colors.qualitative.Plotly,
                   title='Annual Medical Charges')
fig_children.update_layout(bargap=0.1)
fig_children.show()

fig_children = px.violin(medical_data, 
                   x='children',
                   y='charges',
                   color='children', 
                   color_discrete_sequence=px.colors.qualitative.Plotly,
                   title='Annual Medical Charges')
fig_children.update_layout(bargap=0.1)
fig_children.show()

## correlation matrix
num_corr_matrix = medical_data.corr(method = 'pearson' , numeric_only = True)
corr_heatmap=sns.heatmap(num_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
#plt.title('Correlation Matrix', fontsize=16)
#plt.show()


##assessing the Correlation between categorical variables and the target variable 


model = ols('charges ~ C(smoker) + C(region) + C(sex)', data=medical_data).fit()
anova_table = anova_lm(model)
print(anova_table)
r2 = model.rsquared
print(f'R-squared: {r2:.4f}')


## Feature Selection assessment using wrapper method (RFE) with Linear Regression
clean_data = medical_data.drop(columns=['age_bins'], errors='ignore')
encoded_data = pd.get_dummies(clean_data, columns=['sex', 'smoker', 'region'], drop_first=True , dtype=int)
X = encoded_data.drop(columns=['charges'])  
y = encoded_data['charges']
wrapper_model = RFECV(estimator=LinearRegression(), min_features_to_select=1, cv=5, importance_getter='auto')
wrapper_model.fit(X , y)
selected_features = X.columns[wrapper_model.support_]
print(f'Selected features using RFECV: {selected_features.tolist()}')

### backward elimination using p-values
backward_model = SequentialFeatureSelector(LinearRegression(), direction='backward', scoring='r2', cv=5)
backward_model.fit(X, y)
backward_selected_features = X.columns[backward_model.get_support()]
print(f'Selected features using backward elimination: {backward_selected_features.tolist()}')

### forward selection using p-values
forward_model = SequentialFeatureSelector(LinearRegression(), direction='forward', scoring='r2', cv=5)
forward_model.fit(X, y) 
forward_selected_features = X.columns[forward_model.get_support()]
print(f'Selected features using forward selection: {forward_selected_features.tolist()}')


final_selected_features = list(backward_selected_features)
print(f'Final selected features for the model: {final_selected_features}')
final_data = encoded_data[final_selected_features + ['charges']]
final_data = sm.add_constant(final_data)
reg = ols('charges ~ age + bmi + children +smoker_yes', data=final_data).fit()
anova_table = anova_lm(reg)

print(f'R-squared: {reg.rsquared:.4f}')
print(reg.summary())

fitted_values  = reg.fittedvalues


### assessing regression performance using train-test split

train_data, test_data = train_test_split(final_data, test_size=0.2, random_state=42)
X_train = train_data.drop(columns=['charges'])
y_train = train_data['charges']
X_test = test_data.drop(columns=['charges'])
y_test = test_data['charges']

x_train_const = sm.add_constant(X_train)
reg_train = sm.OLS(y_train, x_train_const).fit()

print(reg_train.summary())
print (f'Training R-squared: {reg_train.rsquared:.4f}')

reg_trained = LinearRegression()
reg_trained.fit(X_train, y_train)
print (f'Training R-squared: {reg_trained.score(X_train, y_train):.4f}')

reg_test = LinearRegression()
reg_test_score = reg_trained.score(X_test, y_test)
print (f'Test R-squared: {reg_test_score:.4f}')


##### assessing KNN performance using cross-validation
knn_model = KNeighborsRegressor(n_neighbors=5)
KNN_trained = knn_model.fit(X_train, y_train)
knn_cv_scores = cross_val_score(KNN_trained, X_train, y_train, cv=5, scoring='r2')
print(f'KNN Cross-Validation R-squared scores: {knn_cv_scores}')
print(f'KNN Mean Cross-Validation R-squared: {knn_cv_scores.mean():.4f}')


knn_test_score = KNN_trained.score(X_test, y_test)
print(f'KNN Test R-squared: {knn_test_score:.4f}')
