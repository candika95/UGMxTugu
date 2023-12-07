import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion, default_converter
from scipy.stats import multivariate_normal
from rpy2 import rinterface

def rpy2_init_simple():
    rinterface.initr_simple()

def test_threading_simple():
    from threading import Thread
    thread = Thread(target=rpy2_init_simple)
    thread.start()
test_threading_simple()

import streamlit as st
import numpy as np
from plotly import graph_objs as go
import random



# if threading.current_thread() is threading.main_thread():
#     signal.signal(signal.SIGINT, _sigint_handler)
import os
if 'R_HOME' not in os.environ:
    os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'# C:\Users\ACER\AppData\Local\Temp\Rtmp8uE8aZ\downloaded_packages
# ro.r('''
#     install.packages("sn")
#     library(sn)
# ''')
import rpy2.robjects.numpy2ri
from rpy2.robjects import numpy2ri
numpy2ri.activate()
# import anndata2ri
# anndata2ri.activate()
rpy2.robjects.numpy2ri.activate()
# pbmc = sc.datasets.pbmc68k_reduced()
# ro.globalenv['sce'] = pbmc
# Fungsi untuk menghitung BIC dan AIC
def calculateBICnAIC(ll, y):
  npar = y.shape[1] + sum(range(1, y.shape[1] + 1))
  ll = ll
  nll = ll * (-1)
  n_multi = y.shape[0]
  aic = 2 * npar - 2 * ll
  bic = np.log(n_multi) * npar - 2 * ll
  return bic, aic
# nilai_aic= aicbic(Logl, data_claim)
# def calculateBICnAIC(x, y):
#     npar = y.shape[1] + sum(range(1, y.shape[1] + 1))
#     ll = x['logL']
#     # ll = x.rx2('logL')
#     # nll = ll * (-1)
#     n_multi = y.shape[0]
#     ll_list = list(ll)
#     aic_multi = 2 * npar - 2 * np.array(ll_list)
#     # aic_multi = 2 * npar - 2 * ll
    
#     bic_multi = np.log(np.array(n_multi)) * npar - 2 * ll_list
#     # bic_multi = np.log(n_multi) * npar - 2 * ll
#     # df = pd.DataFrame({ 'aic.multi': [aic_multi], 'bic.multi': [bic_multi]})
#     return bic_multi, aic_multi
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
# Impor paket 'sn'
sn = importr('sn')
# @st.cache_data
def trainSkew(df):
  # with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
  #     # Konversi DataFrame pandas ke DataFrame R
  #     data_claim_r = pandas2ri.py2rpy(df)
      # df = spark.createDataFrame(pandas_df)
    # with (ro.default_converter + pandas2ri.converter).context():
      # data_claim_r = ro.conversion.get_conversion().py2rpy(df)
    pandas2ri.activate()
    data_claim_r = pandas2ri.py2rpy(df)
    
    result = sn.msn_mle(y=data_claim_r, opt_method="BFGS")
    # miu = result['dp']['beta']
    # omega = result['dp']['Omega']
    # alpha = result['dp']['alpha']
    # logL = result['logL']
    miu = result.rx2('dp').rx2('beta')
    omega = result.rx2('dp').rx2('Omega')
    alpha = result.rx2('dp').rx2('alpha')
    logL = result.rx2('logL')
    return result, miu, omega, alpha, logL


# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr
# import rpy2.robjects as robjects

# Fungsi untuk menghitung PDF Skewed Normal
@st.cache_data
def pdf_function(x, mu, omega, alpha):
    pdf_values = np.zeros(x.shape[0])
    m= mu[:,0]
    m= np.repeat(m, x.shape[0])
    n= mu[:,1]
    n= np.repeat(n,x.shape[0])
    mu= pd.DataFrame({'LogNet': m, 'diff_time': n})
    pdf_values += sn.dmsn(x, mu, omega, alpha)
    return pdf_values
# def pdf_function(x, mu, omega, alpha, sn):
#   with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
#     rpy2.robjects.numpy2ri.activate()
#     # sn = importr('sn')
#     pdf_values = np.zeros(x.shape[0])
#     m= mu[:,0]
#     m= np.repeat(m, x.shape[0])
#     n= mu[:,1]
#     n= np.repeat(n,x.shape[0])
#     mu= pd.DataFrame({'LogNet': m, 'diff_time': n})
#     x_list = x.tolist()
#     mu_list = mu.values.tolist()
#     omega_list = omega.tolist()
#     alpha_list = alpha.tolist()

#     # Calculate density values
#     pdf_values += r.sn.dmsn(x_list, mu_list, omega_list, alpha_list)
#   return pdf_values
from rpy2.robjects import numpy2ri, r
from rpy2.robjects import pandas2ri
# Fungsi untuk mengambil sampel dari distribusi Skewed Normal 
@st.cache_data(ttl=3600)
def calculateSample(miuSkew, omegaSkew, alphaSkew):
  np.random.seed(random.randint(0, 1000000))
  sample_size = 100000
  sample = np.zeros((sample_size, 2))
  k= miuSkew[:,0]
  l= miuSkew[:,1]
  mu= pd.DataFrame({'LogNet': k, 'diff_time': l})
  pandas2ri.activate()
  mu_r = pandas2ri.py2rpy(mu)
  # generated_sample = np.zeros((sample_size, 2))
  # mu_r = ro.conversion.py2ri(mu.values)
  for i in range(sample_size):
    # Generate a random sample
    generated_sample = np.transpose(sn.rmsn(1, mu_r, omegaSkew, alphaSkew))
    sample[i, :] = generated_sample.copy()
    # sample[i,:] = np.transpose(sn.rmsn(1, mu_r, omegaSkew, alphaSkew))
  # Sample = pd.DataFrame(sample, columns=['LogNet', 'diff_time'])
  return sample
  # return Sample
# Fungsi untuk menghitung conditional expectation E(C|T) diketahui interval T
@st.cache_data
def conditional_mean(sample, a, b):
    # Filter the sample to include only the rows where lower_limit < T <= upper_limit
    filtered_sample = sample[(sample[:, 1] > a) & (sample[:, 1] <= b), :]
    filtered_sample[:, 0] = np.exp(filtered_sample[:, 0])

    # Calculate expectation
    expectation = np.sum(filtered_sample[:,0]) / len(filtered_sample)

    return expectation
# Fungsi untuk menghitung conditional variance Var(C|T) diketahui interval T
@st.cache_data
def conditional_variance(sample, a, b):
    # Filter the sample to include only the rows where lower_limit < T <= upper_limit
    filtered_sample = sample[(sample[:, 1] > a) & (sample[:, 1] <= b), :]
    filtered_sample[:, 0] = np.exp(filtered_sample[:, 0])

    # Calculate expectation of C and C^2
    expectation_C = np.sum(filtered_sample[:,0]) / len(filtered_sample)
    expectation_C2 = np.sum(filtered_sample[:,0]**2) / len(filtered_sample)

    # Calculate variance
    variance = expectation_C2 - expectation_C**2

    return np.sqrt(variance)
# Fungsi untuk menghitung Value at Risk VaR(C|T) diketahui interval T dengan tingkat kepercayaan alpha tertentu  
@st.cache_data
def calculateVaR(alfa, sample, a, b):
    # Filter the sample to include only the rows where lower_limit < T <= upper_limit
    filtered_sample = sample[(sample[:, 1] > a) & (sample[:, 1] <= b), :]
    filtered_sample[:, 0] = np.exp(filtered_sample[:, 0])

    # Calculate VaR  and TVaR at the given confidence level
    VaR = np.quantile(filtered_sample[:,0], alfa)
    # TVaR = np.mean(filtered_sample[filtered_sample[:, 0] > VaR, 1])
    return VaR
# Fungsi untuk membuat plot Skewed Normal
@st.cache_data
def plotSkew(miuSkew, omegaSkew , alphaSkew, b, alpha):
  results  = []
  results_dates = []
  result_variance = []
  result_var = []
  st.session_state.df['trx_date'] = pd.to_datetime(st.session_state.df['trx_date'])
  start = st.session_state.df['trx_date'].max()
  start_month = start + pd.DateOffset(months=b)
  sample = calculateSample(miuSkew, omegaSkew , alphaSkew)
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
  expectationSkew = np.array(results)
  dates = np.array(results_dates)
  Variance = np.array(result_variance)
  var = np.array(result_var)

  # Create a DataFrame with 'Date' and 'Value' columns
  df = pd.DataFrame({'date': dates, 'Expectation': expectationSkew, 'stDev': Variance, 'VaR':  var})
  traceSkew = go.Bar(
        x = df['date'],
        y = df['Expectation'],
        name="Prediction",
        marker=dict(color='#9100FE'),
        # text=['Expectation']
        )
  figSkew = go.Figure()
  figSkew.add_trace(traceSkew)
  figSkew.update_layout(
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
  figSkew.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
  return df, figSkew


