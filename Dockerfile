FROM tensorflow/tensorflow:2.4.0-jupyter

EXPOSE 8501

RUN apt-get update && yes | apt-get upgrade
RUN apt install -y graphviz sudo

WORKDIR data_science
 
RUN pip install -U pip

# COPY . . 

COPY requirements.txt . 

RUN pip install -r requirements.txt

RUN pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
RUN pip install -U pandas-profiling[notebook]

RUN pip install ipywidgets

RUN pip install pydot
# RUN pip install nbconvert nbconvert[webpdf]

RUN jupyter nbextension enable --py widgetsnbextension

# CMD streamlit run streamlit_app.py