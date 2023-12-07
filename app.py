import streamlit as st
st.set_page_config(
  page_title = 'UGMxTugu',
  page_icon = '📊',
  layout= 'wide',
  initial_sidebar_state= 'expanded'
)
from models import GMM
from models import copula
from models import skew

import numpy as np
from numerize.numerize import numerize
from PIL import Image
import pandas as pd
from plotly import graph_objs as go
import json
from streamlit_lottie import st_lottie
from scipy.stats import multivariate_normal, norm
import xlsxwriter

# import rpy2.robjects as rox 
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
numpy2ri.activate()
# from rpy2.robjects.packages import importr
# from rpy2 import rinterface

import os
if 'R_HOME' not in os.environ:
    os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'


# styling
hide_st_style = """
            <style>
            div.block-container{padding-top: 1rem;}
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)  
# import gambar
tuguIcon = Image.open('./images/Tugu.png')
ugmIcon = Image.open('./images/LogoUgmHorizontal.png')
kedairekaIcon = Image.open('./images/Kedaireka.png')
# mengelola input excel user
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
@st.cache_data(ttl=(60*60*24))
def get_data_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df
@st.cache_data(ttl=(60*60*24))
def get_data_from_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    return df
def try_read_df(f):
    try:
        return get_data_from_csv(f)
    except:
        return get_data_from_excel(f)
# home page
def inputPage():
    with open("./styles/styleInput2.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c3:
        st.image(ugmIcon)
    with c4:
        st.image(tuguIcon)
    with c5:
        st.image(kedairekaIcon)

    markdown_text = """
    <div style="column-count: 1;">
    Welcome to the IFRS 17 model development dashboard as a result of the <span style="font-weight:bold">collaboration<span/>

    between Universitas Gadjah Mada and Tugu Insurance in Kedaireka Matching Fund 2023.
    </div>
    """
    st.markdown(markdown_text, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(" ", type=["csv", "xlsx"])
    df = None  # Initialize df to None
    if uploaded_file is not None:
    #   st.success("File uploaded successfully!")
        # Load Excel data once the file is uploaded
        df = try_read_df(uploaded_file)
        # Save DataFrame to Session State
        st.session_state.df = df
        # Set a session state variable to signal the transition
        st.session_state.transition_to_main = True
        # transition_thread = threading.Thread(target=transition_to_main)
        # transition_thread.start()
        st.rerun()
# fungsi memanggil lottie
@st.cache_data
def load_lottie_json(json_path: str):
    try:
        with open(json_path, "r") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading Lottie JSON: {e}")
        return None
# fungsi seleksi data
def sidebarFilter(df):
    # st.header("Please Filter Here:")
    conditions = []
    model = st.sidebar.selectbox('Model', ('Gaussian Mixture', 'Skewed Normal'))
    timeInterval = st.sidebar.number_input('Time Interval', value=12)
    alpha = st.sidebar.number_input('Alpha', 0.0, 1.0, (0.75), step=0.1)
    cob = None  # Initialize cob to None

    if 'COB' in df.columns:
        cob_options = df["COB"].unique()
        cob = st.sidebar.selectbox(
        "Select the COB:",
        options=cob_options,
    )
        conditions_cob = df["COB"] == cob
    else:
        conditions_cob = pd.Series(True, index=df.index)  # True for all rows if 'COB' column is not present

    if 'Product_Name' in df.columns:
        product_options = df.loc[conditions_cob, "Product_Name"].unique()
        product_options_str = [f"{option}" for option in product_options]
        product_names = st.sidebar.multiselect(
            "Select the Product Name:",
            options=product_options_str,
            default=product_options_str
        )

        if product_names:
            conditions_product = df["Product_Name"].isin(product_names)
            conditions = conditions_cob & conditions_product
            df_selection = df[conditions]
        else:
            df_selection = df[conditions_cob]  # No product names selected, filter based on COB only
    else:
        df_selection = df[conditions_cob]  # No 'Product_Name' column, filter based on COB only

    return df_selection, model, timeInterval, alpha
# mengelola data input sesuai model
@st.cache_data
def preModelGMM(df):
    # df = df.drop_duplicates(subset=['claim_size'])
    # df.dropna(inplace=True)
    # df = DF.dropna()
    df.dropna(inplace=True)
    df = df[df['claim_size'] > 0] # 678
    df['LogNet'] = np.log(df['claim_size'])
    new_cols = ['diff_time', 'LogNet']
    df=df.reindex(columns=new_cols)
    df = df.reset_index(drop=True)
    return df
@st.cache_data
def preModelCopula(df):
    # df = df.drop_duplicates(subset=['claim_size'])
    df.dropna(inplace=True)
    df = df[df['claim_size'] > 0] # 678
    new_cols = ['diff_time', 'claim_size']
    df=df.reindex(columns=new_cols)
    df = df.reset_index(drop=True)
    return df
@st.cache_data
def preModelSkew(df):
    # df = df.drop_duplicates(subset=['claim_size'])
    df.dropna(inplace=True)
    df = df[df['claim_size'] > 0] # 678
    df['LogNet'] = np.log(df['claim_size'])
    new_cols = ['LogNet', 'diff_time']
    df=df.reindex(columns=new_cols)
    df = df.reset_index(drop=True)
    return df
# mengubah dataframe menjadi file excel
def to_excel(df):
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter('data.xlsx', engine='xlsxwriter')

    # Convert the DataFrame to an XlsxWriter Excel object
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # Close the Pandas Excel writer to save the file
    writer.close()
# membuat 3d plot gmm
@st.cache_data
def plot3dGMM(df, index):
    # Buat bins untuk data Anda
    x_c, x_bin_edges = np.histogram(df['diff_time'], bins=20)
    y_c, y_bin_edges = np.histogram(df['LogNet'], bins=20)
    # Hitung frekuensi gabungan pada level potongan
    hist, xedges, yedges = np.histogram2d(df['diff_time'], df['LogNet'], bins=[x_bin_edges, y_bin_edges])
    # Buat grid koordinat (x,y)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    # Buat dimensi bar
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.flatten()
    # Hitung total frekuensi
    total_freq = np.sum(dz)
    # Ubah frekuensi menjadi probabilitas
    dz = dz / total_freq
    
    gmm = GMM.trainGmm(df, index)
    X, _ = gmm.sample(100000)  # mengambil sampel dari model
    # Hitung nilai f(x, y) untuk setiap sampel
    f_values = np.zeros(X.shape[0])
    for i in range(gmm.n_components):
        f_values += gmm.weights_[i] * multivariate_normal(mean=gmm.means_[i], cov=gmm.covariances_[i]).pdf(X)
    # Buat plot 3D dengan Plotly
    fig3dGMM = go.Figure()
    # Tambahkan histogram 3D ke plot
    fig3dGMM.add_trace(go.Mesh3d(x=xpos, y=ypos, z=dz, color='rgba(255, 0, 0, 0.7)', opacity = 0.5,name='Distribution'))
    # Tambahkan scatter plot ke plot
    fig3dGMM.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=f_values, mode='markers', marker=dict(size=2, color='blue'), name='Sample'))
    # Tambahkan label sumbu
    fig3dGMM.update_layout(scene = dict(
                                xaxis_title='Time Difference',
                                yaxis_title='LogNet',
                                zaxis_title='f(c, t)'),
                                title='3D Plot Data vs GMM',
                                width=700,
                                margin=dict(r=20, l=10, b=10, t=40),
                                legend=dict(traceorder='reversed'), showlegend=True)
    return fig3dGMM
# membuat tabel plot gmm
@st.cache_data
def plotTableGMM(df, alpha):
    alpha = alpha * 100
    figtGMM = go.Figure(data=[go.Table(
        columnwidth=[19, 50, 45, 45, 25],
        header=dict(values=['Date', 'Expectation', 'StDev', f"VaR {alpha}%", 'Ratio(%)'],
                    fill_color='#9100FE',
                    font=dict(color='white', size=15)),
        cells=dict(values=[df.date, df.Expectation, df.stDev, df.VaR, df.Ratio],
                   fill_color='white',
                   font=dict(size=15),
                   height=30))])
    figtGMM.update_layout(
    title_text='Table Forecast',
)
    return figtGMM
# membuat tabel plot skew
@st.cache_data
def plotTableSkew(df, alpha):
    alpha = alpha * 100
    figtSkew = go.Figure(data=[go.Table(
        columnwidth=[20, 45, 45, 50, 25],
        header=dict(values=['Date', 'Expectation', 'StDev', f"VaR {alpha}%", 'Ratio(%)'],
                    fill_color='#9100FE',
                    font=dict(color='white', size=15)),
        cells=dict(values=[df.date, df.Expectation, df.stDev, df.VaR, df.Ratio],
                   fill_color='white',
                   font=dict(size=15),
                   height=30))])
    figtSkew.update_layout(
    title_text='Table Forecast',
)
    return figtSkew
# membuat 3d plot skew
@st.cache_data
def plot3dSkew(df, miuSkew, omegaSkew , alphaSkew):
    x_c, x_bin_edges = np.histogram(df['diff_time'], bins=20)
    y_c, y_bin_edges = np.histogram(df['LogNet'], bins=20)

    hist, xedges, yedges = np.histogram2d(df['diff_time'], df['LogNet'], bins=[x_bin_edges, y_bin_edges])

    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.flatten()

    total_freq = np.sum(dz)
    dz = dz / total_freq
    pandas2ri.activate()
    joint_msn = skew.pdf_function(df, miuSkew, omegaSkew , alphaSkew)
    #Membuat data frame
    df2= pd.DataFrame(joint_msn, columns=['joint_msn'])
    #Memasukkan ke dataframe
    data_claim2 = pd.DataFrame({'LogNet': df['LogNet'], 'diff_time': df['diff_time'], "joint_msn" : df2['joint_msn']})
    fig3dSkew = go.Figure()
    fig3dSkew.add_trace(go.Scatter3d(x=data_claim2['diff_time'], y=data_claim2['LogNet'], z=data_claim2['joint_msn'],mode='markers', marker=dict(color='blue', size=2), name='sample'   ))
    fig3dSkew.add_trace(go.Mesh3d(x=xpos, y=ypos, z=dz, color='rgba(255, 0, 0, 0.7)', opacity=0.50))
    fig3dSkew.update_layout(scene=dict(
                                xaxis_title='diff_time',
                                yaxis_title='LogNet',
                                zaxis_title='f(c,t)'),
                                title='3D Plot Data vs Skewed Normal',
                                width=700,
                                margin=dict(r=20, l=10, b=10, t=40),
                                legend=dict(traceorder='reversed'), showlegend=True)
    return fig3dSkew
# page utama yang ditampilkan
def main_page():
    with open("./styles/style.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
    st.header('Best Estimation Value')
    with st.sidebar:
        st_lottie(lottie_json, width=220)  
    df_selection, model, timeInterval, alpha = sidebarFilter(st.session_state.df)
    # st.write(df_selection)
    df_modelGmm = preModelGMM(df_selection)
    # st.write(df_selection)
    bicGMM, iclGMM, indexBicGMM = GMM.calculateBICnICL(df_modelGmm)
    # st.write(indexBicGMM)
    gmm = GMM.trainGmm(df_modelGmm, indexBicGMM) 

    
    df_modelSkew = preModelSkew(df_selection)
    resultSkew, miuSkew, omegaSkew, alphaSkew, logL = skew.trainSkew(df_modelSkew)
    bicSkew, aicSkew = skew.calculateBICnAIC(logL, df_modelSkew)
    
    # bicSkew, aicSkew = skew.calculateBICnICL(result, df_modelSkew)
    # st.write(bicSkew)
    # varGMM = GMM.calculateVaR(0.75, indexBicGMM, gmm.weights_, gmm.means_, gmm.covariances_, 0, timeInterval)
    sampleGMM = GMM.calculateSample(indexBicGMM, gmm.weights_, gmm.means_, gmm.covariances_)
    sampleSkew = skew.calculateSample(miuSkew, omegaSkew, alphaSkew)
    expectationGMM = GMM.conditional_mean(sampleGMM, 0, 12)
    expectationSkew = skew.conditional_mean(sampleSkew, 0, 12)
    dfGMM, figGMM = GMM.plotGMM(indexBicGMM, gmm.weights_, gmm.means_, gmm.covariances_, timeInterval, alpha)
    dfSkew, figSkew = skew.plotSkew(miuSkew, omegaSkew , alphaSkew, timeInterval, alpha)
    models = [
            {
            'name': 'Gaussian Mixture',
            'bic': round(bicGMM, 2),
            'est':expectationGMM
            }
            ,{
            'name': 'Frank Copula',
            'bic': round(9000, 2),
            'est': expectationGMM
        },{
            'name': 'Skewed Normal',
            'bic': np.round(bicSkew[0], 2),
            # 'bic': round(9001, 2),
            'est': expectationSkew
        }]
    best_model = min(models, key=lambda model: model['bic'])
    best_model_name = best_model['name']
    best_model_est = best_model['est']
    totalClaim = df_selection['claim_size'].sum()
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric( label='Total Claim', value=f"{numerize(totalClaim)}")
    with kpi2:                                                  
        st.metric( label='Upcoming Year Estimate', value=f"{numerize(best_model_est)}")
    with kpi3:
        st.metric(label="Best Model", value=f"{best_model_name}")

    i = "BIC"
    sline = '23'
    
    r3 = f"""<p style='background-color: #9100FE; 
                            color: white; 
                            font-size: 18px; 
                            border-radius: 7px; 
                            padding: 25px;
                            width: 200px;
                            margin-left: 20px;
                            margin-top:30px;
                            box-shadow: 0px 5px 10px 0px rgba(0, 0, 0, 0.5);  
                            line-height:25px;'>
                            <span style='font-weight: bold;'>{i}</span>
                            </style><BR><span 
                            style='font-size: 14px;
                            margin-top: 0;'>{sline}</style></span></p>"""
    z1, z2 = st.columns([2,3])
    if model == 'Gaussian Mixture':
        with z1:
            st.subheader('Gaussian Mixture')
            r1GMM = f"""<p style='background-color: #9100FE; 
                            color: white; 
                            font-size: 18px; 
                            border-radius: 7px; 
                            padding: 25px;
                            width: 200px;
                            margin-left: 0px;
                            margin-top:30px;  
                            box-shadow: 0px 10px 20px 2px rgba(0, 0, 0, 0.5);
                            line-height:25px;'>
                            <span style='font-weight: bold;'>{i}</span>
                            </style><BR><span 
                            style='font-size: 14px;
                            margin-top: 0;'>{models[0]['bic']}</style></span></p>"""
            r2GMM = f"""<p style='background-color: #ffa300; 
                            color: white; 
                            font-size: 18px; 
                            border-radius: 7px; 
                            padding: 25px;
                            width: 300px;
                            margin-left: 0px;
                            margin-top:10px;
                            box-shadow: 0px 5px 10px 0px rgba(0, 0, 0, 0.5);  
                            line-height:25px;'>
                            <span style='font-weight: bold;'></span>
                            </style><span 
                            style='
                            font-size: 14px;
                            margin-top: 0;
                            '>Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.</style></span></p>"""
            
            # st.dataframe(df.set_index(df.columns[0]))
            st.markdown(r2GMM, unsafe_allow_html=True)
            st.markdown(r1GMM, unsafe_allow_html=True)
            st.markdown("""
                        <div class='padding-table'></div>""", unsafe_allow_html=True)
            
        with z2:
            fig3dGMM = plot3dGMM(df_modelGmm, indexBicGMM)
            st.plotly_chart(fig3dGMM, use_container_width=True)
        
        dfGMM['Ratio'] = dfGMM['VaR'] / dfGMM['Expectation']
        dfGMM['Ratio'] = dfGMM['Ratio'] * 100
        dfGMM['Ratio'] = dfGMM['Ratio'].round(2)
        # dfGMM['Ratio'] = dfGMM['Ratio'].apply(lambda x: '{:.%}'.format(x))
        dfGMM['Expectation'] = dfGMM['Expectation'].round(0)
        dfGMM['stDev'] = dfGMM['stDev'].round(0)
        dfGMM['VaR'] = dfGMM['VaR'].round(2)
        dfGMM['date'] = dfGMM['date'].dt.strftime('%b %Y')
        dfGMM['Expectation'] = dfGMM['Expectation'].apply(lambda x: 'Rp{:,.0f}'.format(x))
        dfGMM['stDev'] = dfGMM['stDev'].apply(lambda x: '{:,.0f}'.format(x))
        dfGMM['VaR'] = dfGMM['VaR'].apply(lambda x: '{:,.0f}'.format(x))
        
        y1, y2 = st.columns(2)  #[7,6]
        with y1:
            figtGMM = plotTableGMM(dfGMM, alpha)
            st.plotly_chart(figtGMM, use_container_width=True, config={'displayModeBar': False, 'margin': {'t': 0}})
            to_excel(dfGMM)
            with open('data.xlsx', 'rb') as f:
                file_content = f.read()
            st.download_button(label='📥 Download Current Result',
                   data=file_content,
                   key='download_button',
                   file_name='Gaussian-Mixture-Prediction.xlsx',
                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        with y2:
            st.plotly_chart(figGMM, use_container_width=True)
               
    if model == 'Copula':
        with z1:
            st.subheader('Frank Copula')
            r1Frank = f"""<p style='background-color: #9100FE; 
                            color: white; 
                            font-size: 18px; 
                            border-radius: 7px; 
                            padding: 25px;
                            width: 200px;
                            margin-left: 20px;
                            margin-top:30px;  
                            box-shadow: 0px 10px 20px 2px rgba(0, 0, 0, 0.5);
                            line-height:25px;'>
                            <span style='font-weight: bold;'>{i}</span>
                            </style><BR><span 
                            style='font-size: 14px;
                            margin-top: 0;'>{models[1]['bic']}</style></span></p>"""
            r2Frank = f"""<p style='background-color: #ffa300; 
                            color: white; 
                            font-size: 18px; 
                            border-radius: 7px; 
                            padding: 25px;
                            width: 200px;
                            margin-left: 20px;
                            margin-top:30px;
                            box-shadow: 0px 5px 10px 0px rgba(0, 0, 0, 0.5);  
                            line-height:25px;'>
                            <span style='font-weight: bold;'>Variance</span>
                            </style><BR><span 
                            style='font-size: 14px;
                            margin-top: 0;'>{iclGMM}</style></span></p>"""
            st.markdown(r1Frank, unsafe_allow_html=True)
            st.markdown(r2Frank, unsafe_allow_html=True)
            # st.markdown(r3, unsafe_allow_html=True)
        with z2:
            st.write('')
            # Example usage:
            # Generating fake forecast data for demonstration
            # forecast_data = generate_fake_forecast('2023-01-01', '2023-12-31')
            # # Plotting datasets with the fake forecast data
            # fig = plot_datasets(forecast_data)
            # st.plotly_chart(fig, use_container_width=True)
    if model == 'Skewed Normal':
        fig3dSkew = plot3dSkew(df_modelSkew, miuSkew, omegaSkew , alphaSkew)
        t1, t2 = st.columns([2,3])
        with t1:
            st.subheader('Skewed Normal')
            r1GMM = f"""<p style='background-color: #9100FE; 
                            color: white; 
                            font-size: 18px; 
                            border-radius: 7px; 
                            padding: 25px;
                            width: 200px;
                            margin-left: 0px;
                            margin-top:30px;  
                            box-shadow: 0px 10px 20px 2px rgba(0, 0, 0, 0.5);
                            line-height:25px;'>
                            <span style='font-weight: bold;'>{i}</span>
                            </style><BR><span 
                            style='font-size: 14px;
                            margin-top: 0;'>{models[2]['bic']}</style></span></p>"""
            r2GMM = f"""<p style='background-color: #ffa300; 
                            color: white; 
                            font-size: 18px; 
                            border-radius: 7px; 
                            padding: 25px;
                            width: 300px;
                            margin-left: 0px;
                            margin-top:10px;
                            box-shadow: 0px 5px 10px 0px rgba(0, 0, 0, 0.5);  
                            line-height:25px;'>
                            <span style='font-weight: bold;'></span>
                            </style><span 
                            style='
                            font-size: 14px;
                            margin-top: 0;
                            '>Skewed Normal distribution is a continuous probability distribution that generalises the normal distribution to allow for non-zero skewness.</style></span></p>"""
            
            # st.dataframe(df.set_index(df.columns[0]))
            st.markdown(r2GMM, unsafe_allow_html=True)
            st.markdown(r1GMM, unsafe_allow_html=True)
            st.markdown("""
                        <div class='padding-table'></div>""", unsafe_allow_html=True)
        with t2:
            st.plotly_chart(fig3dSkew, use_container_width=True)
            
        # dfSkew, traceSkew = skew.plotSkew(miuSkew, omegaSkew , alphaSkew, timeInterval, alpha, sn)
        dfSkew['Ratio'] = dfSkew['VaR'] / dfSkew['Expectation']
        dfSkew['Ratio'] = dfSkew['Ratio'] * 100
        dfSkew['Ratio'] = dfSkew['Ratio'].round(2)
        # dfSkew['Ratio'] = dfSkew['Ratio'].apply(lambda x: '{:.%}'.format(x))
        dfSkew['Expectation'] = dfSkew['Expectation'].round(0)
        dfSkew['stDev'] = dfSkew['stDev'].round(0)
        dfSkew['VaR'] = dfSkew['VaR'].round(2)
        dfSkew['date'] = dfSkew['date'].dt.strftime('%b %Y')
        dfSkew['Expectation'] = dfSkew['Expectation'].apply(lambda x: 'Rp{:,.0f}'.format(x))
        dfSkew['stDev'] = dfSkew['stDev'].apply(lambda x: '{:,.0f}'.format(x))
        dfSkew['VaR'] = dfSkew['VaR'].apply(lambda x: '{:,.0f}'.format(x))
        
        y1, y2 = st.columns(2)   #[7,6]
        with y1:
            figtSkew = plotTableSkew(dfSkew, alpha)
            st.plotly_chart(figtSkew, use_container_width=True, config={'displayModeBar': False, 'margin': {'t': 0}})
            to_excel(dfSkew)
            with open('data.xlsx', 'rb') as f:
                file_content = f.read()
            st.download_button(label='📥 Download Current Result',
                   data=file_content,
                   key='download_button',
                   file_name='Skewed-Normal-Prediction.xlsx',
                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        with y2:
            st.plotly_chart(figSkew, use_container_width=True)
        
        
    
    # Add a button to go back to the welcome page
    if st.button("Back Home"):
        # Reset the session state variables
        st.session_state.transition_to_main = False
        st.session_state.df = None
        # Trigger a rerun to go back to the welcome page
        st.rerun()
lottie_json = load_lottie_json('images/lottie.json')
# def run_input_page():
#     inputPage()

# def run_main_page():
#     main_page()
# kondisi mengelola 2 page
if __name__ == "__main__":
    if not hasattr(st.session_state, 'transition_to_main') or not st.session_state.transition_to_main:
        inputPage()
    else:
        main_page()
        

