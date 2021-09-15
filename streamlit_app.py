import os
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
import streamlit.components.v1 as components
from streamlit.hashing import _CodeHasher

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

from modules.app_utility import *
from modules.data_utility import *

# ToDo: Reset button
# https://discuss.streamlit.io/t/reseting-the-app-via-a-button/705
# import SessionState
# session = SessionState.get(run_id=0, data=None, columns=None)
# if st.button("Reset"):
#   session.run_id += 1
def main():
    _max_width_()
    state = _get_state()
    pages = {
        "Overview": overview,
        "Pre-Processing": pre_process,
        "Feature Exploration": feature_exploration,
        "Feature Engineering": feature_engineering,
        "Download": download,
        "States": display_state_values
    }

    st.sidebar.header(':floppy_disk: Data')
    _file = st.sidebar.file_uploader("Upload file", type=['csv' 
                                                ,'xlsx'
                                                ,'pickle'])  

    if _file:
        data = get_data(_file)
        index_flag = st.sidebar.checkbox('Reset the index')

        if index_flag:
            index = st.sidebar.selectbox('Choose the index', data.columns)
            if st.sidebar.button('Apply'):
                state.index = pd.to_datetime(data.pop(index))
                state.data = data.set_index(state.index)
        
        else:
            state.data = data

        state.columns = st.sidebar.multiselect('Features', 
                            state.data.columns.tolist(),
                            state.data.columns.tolist())           
    
        st.sidebar.title("Pages")
        page = st.sidebar.radio("Select your page", tuple(pages.keys()))
        # Display the selected page with the session state
        pages[page](state)

    if st.sidebar.button('Reset'):
        state.clear()
    st.sidebar.text('By Shayan Dadman.')
    st.sidebar.text('Uit-Narvik, May 2021.')



    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def display_state_values(state):
    st.write("Input state:", state.data)
    st.write("DF index state:", state.index)
    st.write("Columns state:", state.columns)
    st.write("Checkbox state:", state.checkbox)
    st.write("Selectbox state:", state.selectbox)
    st.write("Multiselect state:", state.multiselect)

def overview(state):
    st.header('Overview')
    explore_data(state.data)
    fig = plot_features(state.data, state.columns)
    st.write(fig)
    # generate pandas profile report
    if st.button('Generate the Report'):
        pr = ProfileReport(state.data, explorative=True).to_html()
        components.html(pr, height=700, width=900, scrolling=True)

def pre_process(state):
    st.header('Pre-Processing')

    st.subheader('Handling Duplicated Values')
    rm_dup = st.checkbox('Remove duplicated values')
    if rm_dup:
        state.data = state.data[~state.data.index.duplicated()].reset_index(drop=True)

    st.subheader('Handling Nan/Missing Values')
    rm_nan = st.checkbox('Remove Nan/Missing values')
    rp_nan = st.checkbox('Replace Nan/Missing values with zero')
    if rm_nan:
        state.data = state.data.dropna().reset_index(drop=True)
    elif rp_nan:
        state.data = state.data.fillna()

    state.data = filter_data(state.data, state.columns)

    transform_data(state.data, state.columns)

    state.data = normalize_data(state.data, state.columns)
    state.data = state.data.set_index(state.index, drop=True)

    st.write(state.data)

def feature_exploration(state):      
    st.write(state.data)
    col1, col2 = st.beta_columns([1, 1]) 
    with col1:
        st.subheader('Correlation Heatmap')
        fig = get_correlation_heatmap(state.data, state.columns)
        st.pyplot(fig)

    with col2:
        st.subheader('Boxplot')
        fig = plot_boxplot(state.data, state.columns, 'h')
        st.pyplot(fig)
    
    st.subheader('Principle Component Analysis')
    try:
        pca_plot(state.data, state.columns)
    except:
        st.write('Failed.')

    st.subheader('Data Feature Impotances')
    st.write(""" Here you can analyze the importance of features within the dataset
                for the prediction model. Select the target feature and the inputs. 
                To carry out the analysis, RandomForestRegressor has been used with 1000 estimators.""")
    # try:
    target_feature = st.selectbox('Select target feature', state.columns)
    input_features = st.multiselect('Select input features', 
                            state.columns,
                            state.columns)
    if st.button('Analyze feature importances'):
        feature_importances, fig = feature_importance_plot(state.data[input_features], 
                                                            target_feature)
        st.pyplot(fig)
    # except:
    #     st.write('Failed.')
    
    st.subheader('Barplot')
    try:
        period = st.selectbox('Select the sampling period', ['Y', 'M', 'W', 'D'], index=1)
        barplot_btn = st.button('Generate', key='barplot_btn')
        if barplot_btn:    
            st.pyplot(get_barplot(state.data, state.columns, period))
    except:
        st.write('Index must be date range.')

    st.subheader('Correlation Matrix')
    try:
        corr_btn = st.button('Generate', key='corr_btn')
        if corr_btn:
            fig = get_pairplot(state.data, state.columns)
            st.pyplot(fig)
    except:
        st.write('Failed to generate the plots.')

    st.subheader('Auto-Correlation and Parial Auto-Correlation Functions')
    try:
        lag = st.number_input('Enter lag value:', value=12, min_value=0)
        acf_btn = st.button('Generate', key='acf_btn')
        if acf_btn:
            fig = plot_auto_correlation(state.data, state.columns, lag)
            st.pyplot(fig)

    except:
        st.write('Failed to generate plots.')

    st.subheader('Fast Fourier Transform Analysis')
    st.write("""
            Set the time resolution in correspond to the dataset resolution. 
            
            The default value is set to number of hours per year (24*365.2524).
            """)
    try:
        time_res = st.number_input('Time Resolution:', value=24*365.2524)
        fft_btn = st.button('Generate', key='fft_btn')
        if fft_btn:
            fig = get_fft_plot(state.data, state.columns, dataset_yearly_resolution=time_res)
            st.pyplot(fig)
    
    except:
        st.write('Failed to generate plots.')


def feature_engineering(state):
    st.header('Feature Engineering')


def download(state):
    st.header('Get the Final Dataset')
    file_format = st.selectbox('File Format', ['csv', 'pickle'])
    if st.button('Download'):
        download_file(state.data, file_format)

class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
            }}
            </style>
            """,
            unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()