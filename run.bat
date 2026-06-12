@echo off
call venv\Scripts\activate
python preprocess.py
streamlit run app.py
