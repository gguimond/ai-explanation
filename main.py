
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from alibi.explainers import PartialDependence, plot_pd, plot_pd_variance
from alibi.explainers import PartialDependenceVariance

import matplotlib.pyplot as plt
#X = range(10)
#plt.plot(X, [x*x for x in X])
#plt.show()

df = pd.read_csv('BostonHousing.csv')
df.head()

# extract feature names
feature_names = df.columns.tolist()
target_feature = 'medv'
feature_names.remove(target_feature)

# define target names
target_names = [target_feature]

#  define categorical columns
categorical_columns_names = []

# define categorical and numerical indices for later preprocessing
categorical_columns_indices = [feature_names.index(cn) for cn in categorical_columns_names]
numerical_columns_indices = [feature_names.index(fn) for fn in feature_names if fn not in categorical_columns_names]

# extract data
X = df[feature_names]
# define target column
y = df[target_feature]

# split data in train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

####################################################


# define and fit the oridnal encoder
oe = OrdinalEncoder().fit(X_train[categorical_columns_names])

# transform the categorical columns to ordinal encoding
X_train.loc[:, categorical_columns_names] = oe.transform(X_train[categorical_columns_names])
X_test.loc[:, categorical_columns_names] = oe.transform(X_test[categorical_columns_names])

# convert data to numpy
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

# define categorical mappings
categorical_names = {i: list(v) for (i, v) in zip(categorical_columns_indices, oe.categories_)}

#######################################################

# define numerical standard sclaer
num_transf = StandardScaler()

# define categorical one-hot encoder
cat_transf = OneHotEncoder(
    categories=[range(len(x)) for x in categorical_names.values()],
    handle_unknown='ignore',
)

# define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transf, categorical_columns_indices),
        ('num', num_transf, numerical_columns_indices),
    ],
    sparse_threshold=0
)

# fit preprocessor
preprocessor.fit(X_train)

# preprocess train and test datasets
X_train_ohe = preprocessor.transform(X_train)
X_test_ohe = preprocessor.transform(X_test)


##########################################Training

# define and fit regressor - feel free to play with the hyperparameters
predictor = RandomForestRegressor(random_state=0)
predictor.fit(X_train_ohe, y_train)

# compute scores
print('Train score: %.2f' % (predictor.score(X_train_ohe, y_train)))
print('Test score: %.2f' % (predictor.score(X_test_ohe, y_test)))

#############################################################Explain

# Creating a Prediction Function - Includes pipeline from preprocessing to prediction
prediction_fn = lambda x: predictor.predict(preprocessor.transform(x))

# TODO: Determine which features are more important
# suggestion: Use PartialDependenceVariance class and explain function using the argument method='importance'

explainer = PartialDependenceVariance(predictor=prediction_fn,
                                        feature_names=feature_names,
                                        categorical_names=categorical_columns_names,
                                        target_names=target_names)
exp_importance = explainer.explain(X=X_train, method='importance') 
# plot the results using 
plot_pd_variance(exp=exp_importance)
plt.show()

explainer2 = PartialDependence(predictor=prediction_fn,
                                        feature_names=feature_names,
                                        categorical_names=categorical_columns_names,
                                        target_names=target_names)
exp_importance2 = explainer2.explain(X=X_train, kind='average') 
plot_pd(exp_importance2)
plt.show()

combined_features = [(feature_names.index("rm"), feature_names.index("age"))]
explainer3 = PartialDependence(predictor=prediction_fn,
                                        feature_names=feature_names,
                                        categorical_names=categorical_columns_names,
                                        target_names=target_names)
exp_importance3 = explainer3.explain(X=X_train, kind='average',features=combined_features) 
plot_pd(exp_importance3)
plt.show()




