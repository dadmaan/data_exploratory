import streamlit as st
import pandas as pd
import base64
from modules.data_utility import *

def convert_data_types(df, types, new_types):
    """ Type conversion for the dataset's features."""
    for i, col in enumerate(df.columns):
        new_type = types[new_types[i]]
        if new_type:
            try:
                df[col] = df[col].astype(new_type)
            except:
                st.write('Could not convert', col, 'to', new_types[i])

def download_file(df, extension):
    # csv
    if extension == 'csv': 
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()    
    # pickle
    else: 
        b = io.BytesIO()
        pickle.dump(df, b)
        b64 = base64.b64encode(b.getvalue()).decode()  
        
    # download link
    href = f'<a href="data:file/csv;base64,{b64}" download="new_file.{extension}">Download {extension}</a>'  
    st.sidebar.write(href, unsafe_allow_html=True)

def explore_data(df):
    """ Get the summary of dataset."""
    st.write('Data:')
    frac = st.slider('Use the slider to randomly sample the data (%)', 1, 100, 100)
    if frac < 100:
        df = df.sample(frac=frac/100)
    st.write(df)  
    # SUMMARY
    df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
    numerical_cols = df_types[~df_types['Data Type'].isin(['object',
                    'bool'])].index.values  
    df_types['Count'] = df.count()
    df_types['Unique Values'] = df.nunique()
    df_types['Min'] = df[numerical_cols].min()
    df_types['Max'] = df[numerical_cols].max()
    df_types['Average'] = df[numerical_cols].mean()
    df_types['Median'] = df[numerical_cols].median()
    df_types['St. Dev.'] = df[numerical_cols].std()  
    
    st.write('Summary:')
    st.write(df_types)


def transform_data(df, columns):
    """ Sample and transform given DataFrame."""
    st.subheader('Data Tranformation')
    st.write("""
            Convert the features data types.
            """)
    cols = columns
    df = df[cols]

    types = {'-':None
           ,'Boolean': '?'
           ,'Byte': 'b'
           ,'Integer':'i'
           ,'Floating point': 'f' 
           ,'Date Time': 'M'
           ,'Time': 'm'
           ,'Unicode String':'U'
           ,'Object': 'O'}
    new_types = {}  
    expander_types = st.beta_expander('Convert Data Types')

    for i, col in enumerate(df.columns):
        txt = 'Convert {} from {} to:'.format(col, df[col].dtypes)
        expander_types.markdown(txt, unsafe_allow_html=True)
        new_types[i] = expander_types.selectbox('Field to be converted:', [*types], index=0, key=i)
    
    st.text(" \n") #break line  
    if st.button('Apply Conversion'):
        return

def apply_filter(df, queries):
    for key, val in queries.items():
        df = df[(df[key] > val[0]) & (df[key] < val[1])]
    return df


def get_data(file):
    """ Fetch DataFrame file with streamlit."""
    extension = file.name.split('.')[1]  
    
    if extension.upper() == 'CSV':
        df = pd.read_csv(file)
    elif extension.upper() == 'XLSX':
        df = pd.read_excel(file, engine='openpyxl')
    elif extension.upper() == 'PICKLE':
        df = pd.read_pickle(file)  

    return df

def normalize_data(df, columns):

    st.subheader('Data Normalization')
    st.write("""
            Use these common methods to see the effect of normalization on the 
            corresponding dataset. Eventually, press the button, in case you 
            want to apply the changes.
            """)
    method = st.radio('Methods',
            ['Standard Scaler', 'Robust Scaler', 'MinMax Scaler'], index=0)
    data = df[columns]

    if method == 'Standard Scaler':
        data_norm, fig = standard_scaler(data)
        st.pyplot(fig)

    elif method == 'Robust Scaler':
        data_norm, fig = robust_scaler(data)
        st.pyplot(fig)

    elif method == 'MinMax Scaler':
        data_norm, fig = minmax_scaler(data)
        st.pyplot(fig)
    
    btn_norm = st.button('Apply Normalization')
    if btn_norm:
        return data_norm
    else:
        return df

def filter_data(df, columns):
        st.subheader('Handling Unusual Values')
        st.write("""Select the corresponding feature from the dataset and filter the values by 
                    defining the lower and upper threshold.""")

        data = df
        col1, col2, col3 = st.beta_columns([1, 1, 1])
        with col1:
            feature = st.selectbox('Feature', columns)
        with col2:
            lower = st.number_input('Enter lower threshold', value=data[feature].min())
        with col3:
            upper = st.number_input('Enter upper threshold', value=data[feature].max())

        filter_btn = st.button('Apply the Filter')

        if filter_btn:
            filter_data = data[(data[feature] > lower) & (data[feature] < upper)]
            st.write(filter_data)
            return filter_data
        else:
            return df

# @st.cache(suppress_st_warning=True)
def pca_plot(df, columns):
    data = df[columns]
    pca, pca_data = get_pca(data)

    col1, col2 = st.beta_columns([1, 1])
    with col1:
        comp1 = st.selectbox('Select component one:', np.arange(pca_data.shape[1]), index=0)
    with col2:
        comp2 = st.selectbox('Select component two:', np.arange(pca_data.shape[1]), index=1)

    load_data = pd.DataFrame((pca_data[:,comp1], pca_data[:,comp2])).T
    pca_coeff = pd.DataFrame((pca.components_[comp1,:], pca.components_[comp2,:])).T

    pca_btn = st.button('Generate', key='pca_btn')
    if pca_btn:
        fig = plot_pca_loading(load_data, pca_coeff.values, labels=columns, scale=3)

        st.pyplot(fig)

def feature_importance_plot(df, target_feature, n_trees=1000):
    train_dataset, _, train_labels, _, features_list = get_training_examples(df, target_feature=target_feature)

    rf = RandomForestRegressor(n_estimators=n_trees, random_state=55, n_jobs=-1)
    rf.fit(train_dataset, train_labels)

    return plot_variable_importances(rf, features_list)