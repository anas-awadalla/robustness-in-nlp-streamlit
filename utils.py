import streamlit as st
import numpy as np
import scipy

@st.cache()
def bootstrap_ci(df, x, y, n=1000):
    # A function to calculate the confidence interval for a linear regression model using bootstrap
    # x: independent variable
    # y: dependent variable
    # n: number of bootstrap samples
    # returns: mean of bootstrap
    preds = []
    for _ in range(n):
        bootstrap_df = df.sample(frac=1, replace=True)
        # Add a trendline with confidence intervals using bootstrap
        z = np.polyfit(bootstrap_df['iid_f1_transformed'],
                        bootstrap_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(df['iid_f1_transformed'])
        preds.append(y_fit)

    return np.array(preds).mean(axis=0), preds

def transform(f1, scaling="Linear"):
    if type(f1) is list:
        f1 = np.array(f1)
    # divide by 100 to get the percentage
    f1 = f1 / 100
    if scaling == 'Linear':
        return f1
    elif scaling == 'Probit':
        return scipy.stats.norm.ppf(f1)
    elif scaling == 'Logit':
        return np.log(np.divide(f1, 1.0 - f1))
    