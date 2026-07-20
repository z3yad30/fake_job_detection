import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import html
import nltk
from nltk.corpus import stopwords
from pathlib import Path

st.set_page_config(page_title="Fake Job Detector", page_icon="🔍", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }
    .main-title { font-family: 'Space Mono', monospace; font-size: 2rem; color: #00ff88; margin-bottom: 0; }
    .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 2rem; }
    .result-box-safe { background: #0a2e1a; border: 2px solid #00ff88; border-radius: 12px; padding: 1.5rem; text-align: center; margin-top: 1rem; }
    .result-box-fraud { background: #2e0a0a; border: 2px solid #ff4444; border-radius: 12px; padding: 1.5rem; text-align: center; margin-top: 1rem; }
    .result-label { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; }
    .prob-text { font-size: 0.9rem; color: #aaa; margin-top: 0.5rem; }
    div[data-testid="stSelectbox"] label, div[data-testid="stTextArea"] label, div[data-testid="stTextInput"] label { color: #ccc !important; font-size: 0.85rem !important; }
    .stButton > button { background-color: #00ff88; color: #0f0f0f; font-family: 'Space Mono', monospace; font-weight: 700; border: none; border-radius: 8px; padding: 0.6rem 2rem; width: 100%; font-size: 1rem; }
    .stButton > button:hover { background-color: #00cc6a; }
    .section-header { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #00ff88; letter-spacing: 2px; text-transform: uppercase; margin: 1.5rem 0 0.5rem 0; border-bottom: 1px solid #222; padding-bottom: 0.3rem; }
    .flag-item { font-size: 0.8rem; color: #ff6b6b; padding: 2px 0; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parents[1]
if not (BASE_DIR / "models").exists():
    BASE_DIR = Path.cwd()


@st.cache_resource
def load_artifacts():
    nltk.download('stopwords', quiet=True)
    with open(BASE_DIR / 'models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(BASE_DIR / 'data/processed/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    results_df = pd.read_csv(BASE_DIR / 'results/model_comparison.csv')
    best_model_name = results_df.loc[results_df['f1_score'].idxmax(), 'model']
    model_filename = f"model_{best_model_name.replace(' ', '_').lower()}.pkl"
    with open(BASE_DIR / f'models/{model_filename}', 'rb') as f:
        model = pickle.load(f)
    return scaler, feature_names, model, best_model_name


def normalize_text(text: str) -> str:
    text = '' if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", ' [URL] ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = text.replace('-', ' ')
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_cleaned_text(raw_text: str) -> str:
    nltk.download('stopwords', quiet=True)
    english_stopwords = set(stopwords.words('english'))
    tokens = normalize_text(raw_text).split()
    tokens = [t for t in tokens if t not in english_stopwords and len(t) >= 3]
    return ' '.join(tokens)


TEXT_PATTERNS = {
    'has_wire_transfer':      r'wire\s+transfer|wire\s+funds|send\s+money',
    'has_upfront_payment':    r'upfront|pay\s+first|advance\s+payment',
    'has_unverified_company': r'\bstartup\b|new\s+company|not\s+established|unknown\s+company',
    'has_generic_location':   r'work\s*from\s*home|\bremote\b|\banywhere\b|home\s*based',
    'has_urgency_language':   r'\basap\b|urgent|immediate\s+start|now\s+hiring',
    'has_too_good_promise':   r'easy\s+money|get\s+rich|quick\s+cash|no\s+experience\s+required',
    'has_data_entry_money':   r'data\s+entry.*(commission|payment)|commission.*data\s+entry|paid\s+per\s+entry',
}
GRAMMAR_PATTERN = re.compile(r'([!?.,])\1{2,}', re.IGNORECASE)


def compute_red_flags(consolidated_text: str, has_logo: bool, salary_range: str = '') -> dict:
    flags = {}
    for col, pattern in TEXT_PATTERNS.items():
        flags[col] = int(bool(re.search(pattern, consolidated_text, re.IGNORECASE)))
    flags['has_missing_logo'] = int(not has_logo)
    vague_values = ['not specified', '', 'nan', 'null']
    flags['has_vague_salary'] = int(not salary_range or salary_range.strip().lower() in vague_values)
    punct_count = len(re.findall(r'[!?.,;:]', consolidated_text))
    flags['has_poor_grammar'] = int(bool(GRAMMAR_PATTERN.search(consolidated_text)) or punct_count > 80)
    count = sum(flags.values())
    flags['red_flag_count'] = count
    flags['fraud_risk_score'] = min(count / 5.0, 1.0)
    flags['has_multiple_red_flags'] = int(count >= 3)
    return flags


def parse_bool(value, default=False):
    if pd.isna(value):
        return int(default)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {'1', 'true', 'yes', 'y', 't'}:
            return 1
        if value in {'0', 'false', 'no', 'n', 'f'}:
            return 0
    try:
        return int(bool(value))
    except Exception:
        return int(default)


def prepare_raw_inputs(raw):
    raw = dict(raw) if not isinstance(raw, dict) else raw.copy()
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        value = raw.get(col, '')
        raw[col] = '' if pd.isna(value) else str(value)

    categorical_defaults = {
        'salary_range': 'Not Specified',
        'employment_type': 'Not Specified',
        'required_experience': 'Not Specified',
        'required_education': 'Not Specified',
        'industry': 'Not Specified',
        'function': 'Not Specified'
    }
    for col, default in categorical_defaults.items():
        value = raw.get(col, default)
        if pd.isna(value) or str(value).strip() == '':
            raw[col] = default
        else:
            raw[col] = str(value)

    raw['telecommuting'] = parse_bool(raw.get('telecommuting', 0), default=False)
    raw['has_company_logo'] = parse_bool(raw.get('has_company_logo', 1), default=True)
    raw['has_questions'] = parse_bool(raw.get('has_questions', 0), default=False)
    raw['salary_range'] = '' if raw.get('salary_range') in ['', 'Not Specified', 'nan', 'None'] else raw.get('salary_range')

    return raw


def build_feature_vector(inputs, feature_names):
    inputs = prepare_raw_inputs(inputs)
    consolidated_text = ' '.join([
        inputs.get('title', ''), inputs.get('company_profile', ''),
        inputs.get('description', ''), inputs.get('requirements', ''),
        inputs.get('benefits', ''),
    ])
    cleaned_text = get_cleaned_text(consolidated_text)

    tfidf_features = {f: 0.0 for f in feature_names if f.startswith('tfidf_')}
    words = cleaned_text.split()
    total = len(words) if words else 1
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    for word, count in counts.items():
        key = f'tfidf_{word}'
        if key in tfidf_features:
            tfidf_features[key] = count / total

    red_flags = compute_red_flags(consolidated_text, bool(inputs.get('has_company_logo', True)), inputs.get('salary_range', ''))

    categorical_maps = {
        'employment_type':     {'Full-time': 0, 'Part-time': 1, 'Contract': 2, 'Temporary': 3, 'Other': 4, 'Not Specified': 5},
        'required_experience': {'Not Applicable': 0, 'Internship': 1, 'Entry level': 2, 'Associate': 3, 'Mid-Senior level': 4, 'Director': 5, 'Executive': 6, 'Not Specified': 7},
        'required_education':  {'Not Specified': 0, 'Unspecified': 0, 'High School': 1, 'Some College': 2, "Bachelor's Degree": 3, "Master's Degree": 4, 'Doctorate': 5, 'Professional': 6},
        'industry':            {'Technology': 0, 'Marketing and Advertising': 1, 'Finance': 2, 'Hospital & Health Care': 3, 'Education': 4, 'Retail': 5, 'Computer Software': 6, 'Other': 7, 'Not Specified': 8},
        'function':            {'Engineering': 0, 'Sales': 1, 'Marketing': 2, 'Finance': 3, 'Administrative': 4, 'Customer Service': 5, 'Health Care Provider': 6, 'Other': 7, 'Not Specified': 8},
    }

    row = {
        'telecommuting':    int(inputs.get('telecommuting', False)),
        'has_company_logo': int(inputs.get('has_company_logo', True)),
        'has_questions':    int(inputs.get('has_questions', False)),
    }
    for col, mapping in categorical_maps.items():
        row[col] = mapping.get(inputs.get(col, 'Not Specified'), 0)

    row.update(tfidf_features)
    row.update(red_flags)

    vector = pd.DataFrame([row])
    for col in feature_names:
        if col not in vector.columns:
            vector[col] = 0
    return vector[feature_names], red_flags


# UI
st.markdown('<p class="main-title">🔍 Fake Job Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter job posting details to check if it\'s legitimate or fraudulent</p>', unsafe_allow_html=True)

try:
    scaler, feature_names, model, best_model_name = load_artifacts()
    st.markdown(f'<p style="font-size:0.75rem;color:#555;">Using model: {best_model_name}</p>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

st.markdown('<p class="section-header">Raw Job Posting Input</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader('Upload raw fake_job_postings.csv', type=['csv'], help='Upload one or more raw job postings in the original dataset format.')
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.markdown('<p class="subtitle">Preview uploaded data</p>', unsafe_allow_html=True)
    st.dataframe(raw_df.head())

    if st.button('Predict Uploaded Postings'):
        raw_df = raw_df.drop(columns=['fraudulent'], errors='ignore')
        predictions = []
        
        st.markdown('<p class="section-header">Uploaded Predictions</p>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        total_rows = len(raw_df)
        for idx, (_, row) in enumerate(raw_df.iterrows()):
            inputs = prepare_raw_inputs(row.to_dict())
            vector, red_flags = build_feature_vector(inputs, feature_names)
            vector_scaled = scaler.transform(vector)
            prediction = int(model.predict(vector_scaled)[0])
            probability = float(model.predict_proba(vector_scaled)[0][1])
            predictions.append({
                'job_id': row.get('job_id', ''),
                'title': row.get('title', ''),
                'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate',
                'fraud_probability': probability,
                'red_flag_count': red_flags.get('red_flag_count', 0),
                'fraud_risk_score': red_flags.get('fraud_risk_score', 0.0),
                'has_multiple_red_flags': red_flags.get('has_multiple_red_flags', 0),
            })
            
            # Update progress in real-time
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing: {idx + 1} of {total_rows} predictions completed...")
            
            # Display results incrementally
            result_df = pd.DataFrame(predictions)
            
            def highlight_rows(row):
                colors = [''] * len(row)
                if row['prediction'] == 'Legitimate' and row['fraud_probability'] < 0.5:
                    colors = ['background-color: #0a3d1a'] * len(row)  # Dark green for safe legitimate
                elif row['prediction'] == 'Legitimate':
                    colors = ['background-color: #1a5c3a'] * len(row)  # Medium green for legitimate
                elif row['prediction'] == 'Fraudulent' and row['fraud_probability'] < 0.5:
                    colors = ['background-color: #2d7a3a'] * len(row)  # Light green for uncertain fraudulent
                elif row['prediction'] == 'Fraudulent' and row['fraud_probability'] >= 0.7:
                    colors = ['background-color: #661a1a'] * len(row)  # Red only for high-confidence fraud
                return colors
            
            styled_df = result_df.style.apply(highlight_rows, axis=1)
            styled_df.set_properties(**{'color': '#f0f0f0', 'text-align': 'left'})
            results_container.dataframe(styled_df, use_container_width=True)
        
        status_text.text(f"✅ Completed! All {total_rows} predictions processed.")
        
        # Summary statistics
        result_df = pd.DataFrame(predictions)
        fraudulent_count = len(result_df[result_df['prediction'] == 'Fraudulent'])
        legitimate_count = len(result_df[result_df['prediction'] == 'Legitimate'])
        
        st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)
        col_summary1, col_summary2 = st.columns(2)
        with col_summary1:
            st.metric("🔴 Fraudulent Postings", fraudulent_count, delta=None)
        with col_summary2:
            st.metric("🟢 Legitimate Postings", legitimate_count, delta=None)

st.markdown('<p class="section-header">Manual Raw Input</p>', unsafe_allow_html=True)
job_id          = st.text_input('Job ID (optional)')
location        = st.text_input('Location', placeholder='e.g. US, NY, New York')
department      = st.text_input('Department', placeholder='e.g. Marketing')
salary_range    = st.text_input('Salary Range', placeholder='e.g. $40,000 - $55,000')
title           = st.text_input('Job Title', placeholder='e.g. Software Engineer')
company_profile = st.text_area('Company Profile', placeholder='Brief description of the company...', height=80)
description     = st.text_area('Description', placeholder='What does the role involve...', height=120)
requirements    = st.text_area('Requirements', placeholder='Skills and qualifications needed...', height=80)
benefits        = st.text_area('Benefits', placeholder='What the company offers...', height=60)

st.markdown('<p class="section-header">Job Details</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    employment_type     = st.selectbox('Employment Type', ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Other', 'Not Specified'])
    required_experience = st.selectbox('Required Experience', ['Not Applicable', 'Internship', 'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive', 'Not Specified'])
    industry            = st.selectbox('Industry', ['Technology', 'Marketing and Advertising', 'Finance', 'Hospital & Health Care', 'Education', 'Retail', 'Computer Software', 'Other', 'Not Specified'])
with col2:
    required_education  = st.selectbox('Required Education', ['Not Specified', 'High School', 'Some College', "Bachelor's Degree", "Master's Degree", 'Doctorate', 'Professional'])
    function            = st.selectbox('Function', ['Engineering', 'Sales', 'Marketing', 'Finance', 'Administrative', 'Customer Service', 'Health Care Provider', 'Other', 'Not Specified'])

st.markdown('<p class="section-header">Extra</p>', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    telecommuting    = st.checkbox('Remote / Telecommuting')
with col4:
    has_company_logo = st.checkbox('Has Company Logo', value=True)
with col5:
    has_questions    = st.checkbox('Has Screening Questions')

st.markdown('<br>', unsafe_allow_html=True)

if st.button('Predict Single Raw Posting'):
    if not title and not description:
        st.warning('Please enter at least a job title or description.')
    else:
        inputs = {
            'job_id': job_id,
            'location': location,
            'department': department,
            'salary_range': salary_range,
            'title': title,
            'company_profile': company_profile,
            'description': description,
            'requirements': requirements,
            'benefits': benefits,
            'employment_type': employment_type,
            'required_experience': required_experience,
            'required_education': required_education,
            'industry': industry,
            'function': function,
            'telecommuting': telecommuting,
            'has_company_logo': has_company_logo,
            'has_questions': has_questions,
        }

        vector, red_flags = build_feature_vector(inputs, feature_names)
        vector_scaled = scaler.transform(vector)
        prediction = int(model.predict(vector_scaled)[0])
        probability = float(model.predict_proba(vector_scaled)[0][1])

        if prediction == 1:
            st.markdown(f"""
            <div class="result-box-fraud">
                <div class="result-label" style="color:#ff4444;">⚠️ FRAUDULENT</div>
                <div class="prob-text">Fraud probability: {probability:.1%}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-safe">
                <div class="result-label" style="color:#00ff88;">✅ LEGITIMATE</div>
                <div class="prob-text">Fraud probability: {probability:.1%}</div>
            </div>""", unsafe_allow_html=True)

        detected_flags = [k for k, v in red_flags.items()
                          if k not in ('red_flag_count', 'fraud_risk_score', 'has_multiple_red_flags') and v == 1]
        if detected_flags:
            st.markdown('<p class="section-header">Red Flags Detected</p>', unsafe_allow_html=True)
            for flag in detected_flags:
                label = flag.replace('has_', '').replace('_', ' ').title()
                st.markdown(f'<div class="flag-item">⚑ {label}</div>', unsafe_allow_html=True)

# 
# cd /d "d:\University\semester_6\Data Mining techniques\Project\fake_job_detection"
# 
# py -m venv .venv
# .venv\Scripts\activate.bat
# 
# py -m pip install --upgrade pip
# pip install streamlit pandas numpy nltk scikit-learn mlxtend imbalanced-learn xgboost matplotlib seaborn joblib
# 
# py -c "import nltk; nltk.download('stopwords')"
# 
# py -m streamlit run Note_Books\07_streamlit.py
# '''