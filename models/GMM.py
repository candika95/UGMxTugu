from plotly import graph_objs as go
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm
import pandas as pd
import streamlit as st
# from datetime import datetime
import random

# Fungsi untuk menghitung BIC dan ICL
@st.cache_data
def calculateBICnICL(df):
  BIC = []
  ICL = []
  for i in range(1, 21):
    gmm = GaussianMixture(n_components=i, random_state=0).fit(df)
    BIC.append(gmm.bic(df))
    # Menghitung ICL
    log_likelihood = gmm.score_samples(df)
    entropy = -np.mean(logsumexp(log_likelihood))
    ICL.append(BIC[-1] + 2 * entropy * len(df))
  indexBic = np.argmin(BIC)
  min_BIC = BIC[indexBic]
  min_ICL = ICL[indexBic]
  indexBic+=1
  return min_BIC, min_ICL, indexBic
@st.cache_data
def trainGmm(df, index):
  gmm = GaussianMixture(n_components=index, covariance_type='full', random_state=0).fit(df)
  return gmm
# Fungsi untuk mengambil sampel random dari model GMM
@st.cache_data
def calculateSample(G, pro, mean, var):
  # Dibuat set seed dari awal supaya hasilnya perhitungan konsisten
  # Namun setiap program diulang, sampel akan berubah
  np.random.seed(random.randint(0, 1000000))

  sample_size = 100000
  sample = np.zeros((sample_size, 2))
  for i in range(sample_size):
    # Choose a component at random based on the weights
    component = np.random.choice(G, size=1, p=pro)[0]

    # Generate a random sample from the chosen component
    sample[i, :] = np.random.multivariate_normal(mean=mean[component,:], cov=var[component,:])
  return sample
  # return pd.DataFrame(sample, columns=['diff_time', 'LogNet'])
# Fungsi untuk menghitung conditional expectation E(C|T) diketahui interval T
@st.cache_data
def conditional_mean(sample, a, b):
  # Filter the sample to include only the rows where lower_limit < T <= upper_limit
    filtered_sample = sample[(sample[:, 0] > a) & (sample[:, 0] <= b), :]
    filtered_sample[:, 1] = np.exp(filtered_sample[:, 1])

    # Calculate expectation
    expectation = np.sum(filtered_sample[:,1]) / len(filtered_sample)

    return expectation
# conditional_mean(5, gmm.weights_, gmm.means_, gmm.covariances_, 0, 12)
# Fungsi untuk menghitung conditional variance Var(C|T) diketahui interval T
@st.cache_data
def conditional_variance(sample, a, b):
  # Filter the sample to include only the rows where lower_limit < T <= upper_limit
  filtered_sample = sample[(sample[:, 0] > a) & (sample[:, 0] <= b), :]
  filtered_sample[:, 1] = np.exp(filtered_sample[:, 1])

  # Calculate expectation of C and C^2
  expectation_C = np.sum(filtered_sample[:,1]) / len(filtered_sample)
  expectation_C2 = np.sum(filtered_sample[:,1]**2) / len(filtered_sample)

  # Calculate variance
  variance = expectation_C2 - expectation_C**2

  return np.sqrt(variance)
    # print(f"Var(C| {a} < T <= {b}) = {variance}")
    # print(f"Stdev(C| {a} < T <= {b}) = {np.sqrt(variance)}")
# conditional_variance(6, gmm.weights_, gmm.means_, gmm.covariances_, 0, 12)
# Fungsi untuk menghitung Value at Risk VaR(C|T) diketahui interval T pada tingkat kepercayaan (alpha) tertentu
@st.cache_data
def calculateVaR(alpha, sample, a, b):
  # Filter the sample to include only the rows where lower_limit < T <= upper_limit
  filtered_sample = sample[(sample[:, 0] > a) & (sample[:, 0] <= b), :]
  filtered_sample[:, 1] = np.exp(filtered_sample[:, 1])

  # Calculate VaR  and TVaR at the given confidence level
  VaR = np.quantile(filtered_sample[:,1], alpha)
  TVaR = np.mean(filtered_sample[filtered_sample[:, 1] > VaR, 1])
  return VaR
    # print(f"VaR at {alpha*100}% confidence level for range {a} to {b} is: {VaR}")
    #print(f"TVaR at {alpha*100}% confidence level for range {a} to {b} is: {TVaR}")
# VaR(0.75, 5, gmm.weights_, gmm.means_, gmm.covariances_, 0, 12)
# Fungsi untuk membuat plot Gaussian Mixture Model
@st.cache_data
def plotGMM(G, pro, mean, var, b, alpha):
  results  = []
  results_dates = []
  result_variance = []
  result_var = []
  st.session_state.df['trx_date'] = pd.to_datetime(st.session_state.df['trx_date'])
  start = st.session_state.df['trx_date'].max()
  start_month = start + pd.DateOffset(months=b)
  # current_date = datetime.now()
  # start_month = pd.to_datetime(current_date.replace(month=12, day=30, hour=23, minute=59, second=59, microsecond=0, year=current_date.year))
  # start_month = pd.to_datetime(datetime.now().strftime('%Y-%m')) #%m
  sample = calculateSample(G, pro, mean, var)
  for i in range(0, 109, b):
    a = i
    end = a + b
    expectation = conditional_mean(sample, a, end)
    results.append(expectation)
    # Generate the corresponding date for the current iteration
    current_date = start_month + pd.DateOffset(months=i)
    results_dates.append(current_date)
    variance = conditional_variance(sample, a, end)
    result_variance.append(variance)
    var = calculateVaR(alpha, sample, a, end)
    result_var.append(var)
  # Convert the lists to NumPy arrays
  expectationGMM = np.array(results)
  dates = np.array(results_dates)
  Variance = np.array(result_variance)
  var = np.array(result_var)

  # Create a DataFrame with 'Date' and 'Value' columns
  df = pd.DataFrame({'date': dates, 'Expectation': expectationGMM, 'stDev': Variance, 'VaR':  var})
  # df['date'] = df['Date'].dt.to_period('M')
  # new_cols = ['date', 'Expectation', 'stdev']
  # df=df.reindex(columns=new_cols)
  # df = df.reset_index(drop=True)
  traceGMM = go.Bar(
        x = df['date'],
        y = df['Expectation'],
        name="Prediction",
        marker=dict(color='#9100FE'),
        # text=['Expectation']
        )
  figGMM = go.Figure()    
  figGMM.add_trace(traceGMM)
  figGMM.update_layout(
      plot_bgcolor='white',  # Set plot background color to white
      paper_bgcolor='white',  # Set paper background color to white
      title='Claim Forecast',
      font=dict(color='rgb(0, 0, 0)', size=17),  # Set font color to black
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1
      ),
      yaxis=dict(
          side='left',
          title='Expectation'),
      xaxis=dict(title='Date'),
      hovermode='x',  # Set hovermode to 'x'
      hoverlabel=dict(
      font_color="white",
      bgcolor="#262f69"))
  figGMM.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
  return df, figGMM
  
    


