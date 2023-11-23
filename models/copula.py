from plotly import graph_objs as go

def frank(df):
  # trace_ppn = go.Scatter(
  #       x = df["trx_date"],
  #       y = df["ppn_amt_tbc"]*-1,
  #       mode = 'lines',
  #       # fill = "tonexty",
  #       name="ppn",
  #       line = dict(color= "#57b8ff"), 
  #       )
  # data = [trace_ppn]
  # layout = go.Layout(
  #   title='Frank Copula', 
  #   hovermode='x', 
  #   xaxis = dict(title = "Date"),
  #   yaxis = dict(title = "PPN"))
  # fig = go.Figure({'data':data,'layout':layout})
  
  totalClaim = 11
  est1Year = 1
  bic = 1.1
  return bic, totalClaim, est1Year, df
