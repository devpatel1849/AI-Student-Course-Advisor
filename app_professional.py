import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= LOAD DATASET =================
@st.cache_data
def load_data():
    return pd.read_csv("student_courses_large.csv")

data = load_data()

# ================= ENCODING =================
le_interest = LabelEncoder()
le_goal = LabelEncoder()
le_course = LabelEncoder()

data["Interest_enc"] = le_interest.fit_transform(data["Interest"])
data["Goal_enc"] = le_goal.fit_transform(data["Goal"])
data["Course_enc"] = le_course.fit_transform(data["Recommended_Course"])

X = data[["Interest_enc", "Percentage", "Goal_enc"]]
y = data["Course_enc"]

# ================= MODEL =================
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

# ================= STREAMLIT UI =================
st.set_page_config(
    page_title="AI Student Course Advisor", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin-top: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Comprehensive Sidebar Styling */
    .css-1d391kg, 
    .css-6qob1r, 
    .css-1lcbmhc,
    .css-17eq0hr,
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.98) !important;
    }
    
    /* All sidebar text elements */
    section[data-testid="stSidebar"] * {
        color: #2c3e50 !important;
    }
    
    /* Sidebar labels and text */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Selectbox styling */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 2px solid #e9ecef !important;
        border-radius: 8px !important;
        color: #2c3e50 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div > div {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    /* Slider styling */
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
    }
    
    section[data-testid="stSidebar"] .stSlider .stMarkdown {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* File uploader styling */
    section[data-testid="stSidebar"] .stFileUploader {
        background: #ffffff !important;
        border: 2px dashed #4facfe !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] .stFileUploader div {
        color: #2c3e50 !important;
    }
    
    /* Main content text colors */
    h1 {
        color: #2c3e50 !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2, h3, h4 {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Tab text colors */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #2c3e50 !important;
    }
    
    /* Metric text colors */
    .css-1xarl3l {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header with professional design
st.markdown("""
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -2rem -2rem 2rem -2rem; border-radius: 0 0 20px 20px;">
    <h1 style="color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🎓 AI Student Course Advisor</h1>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 0.5rem 0 0 0;">Intelligent Course Recommendations Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---- NAVIGATION TABS ----
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Course Recommendation", "📊 Analytics Dashboard", "📈 Model Performance", "🗂️ Dataset Explorer"])

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">🔧 Student Input Panel</h3>
    </div>
    """, unsafe_allow_html=True)
    
    interest = st.selectbox("📚 Select Interest", le_interest.classes_, help="Choose your primary area of interest")
    percentage = st.slider("📊 Academic Percentage (%)", 40, 100, 75, help="Your current academic performance")
    goal = st.selectbox("🚀 Future Goal", le_goal.classes_, help="What do you want to achieve?")
    
    st.markdown("---")
    
    # 📸 Photo Upload
    photo = st.file_uploader("📸 Upload Your Photo", type=["png", "jpg", "jpeg"], help="Optional: Add your photo to the report")
    
    # Quick Stats
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h4 style="color: #2c3e50; margin: 0 0 0.5rem 0;">📋 Quick Stats</h4>
        <p style="margin: 0.2rem 0; color: #34495e;">📊 Dataset Size: {}</p>
        <p style="margin: 0.2rem 0; color: #34495e;">🎯 Model Accuracy: {:.1f}%</p>
        <p style="margin: 0.2rem 0; color: #34495e;">🏷️ Available Courses: {}</p>
    </div>
    """.format(len(data), acc*100, len(le_course.classes_)), unsafe_allow_html=True)

student_report = None  # store recommendation

with tab1:
    st.markdown("### 🎯 Get Your Personalized Course Recommendation")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Get Recommendation", use_container_width=True):
            interest_enc = le_interest.transform([interest])[0]
            goal_enc = le_goal.transform([goal])[0]
            pred = model.predict([[interest_enc, percentage, goal_enc]])[0]
            course = le_course.inverse_transform([pred])[0]

            probs = model.predict_proba([[interest_enc, percentage, goal_enc]])[0]
            top_indices = probs.argsort()[-3:][::-1]
            top_courses = le_course.inverse_transform(top_indices)

            # 🎨 Student Profile Card with Photo
            photo_html = ""
            if photo is not None:
                encoded_img = base64.b64encode(photo.read()).decode()
                photo_html = f'<img src="data:image/png;base64,{encoded_img}" style="width:120px; height:120px; border-radius:50%; display:block; margin:auto; margin-bottom:10px; border: 3px solid white;">'

            # Professional Student Profile Card
            st.markdown("### 🎓 Student Profile Report")
            
            # Create columns for better layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if photo is not None:
                    st.image(photo, width=150, caption="Student Photo")
                else:
                    st.markdown("""
                    <div style="
                        width: 150px; 
                        height: 150px; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 0 auto;
                    ">
                        <span style="font-size: 3rem; color: white;">👤</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    border-left: 5px solid #667eea;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                ">
                    <h4 style="color: #2c3e50; margin-bottom: 1rem;">📋 Student Information</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-bottom: 1rem;">
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #4facfe;">
                            <strong style="color: #667eea;">📚 Interest:</strong><br>
                            <span style="color: #2c3e50; font-size: 1.1rem;">{interest.title()}</span>
                        </div>
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #28a745;">
                            <strong style="color: #28a745;">📊 Percentage:</strong><br>
                            <span style="color: #2c3e50; font-size: 1.1rem;">{percentage}%</span>
                        </div>
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #fd7e14;">
                            <strong style="color: #fd7e14;">🚀 Future Goal:</strong><br>
                            <span style="color: #2c3e50; font-size: 1.1rem;">{goal.title()}</span>
                        </div>
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #6f42c1;">
                            <strong style="color: #6f42c1;">📅 Generated:</strong><br>
                            <span style="color: #2c3e50; font-size: 1.1rem;">{datetime.now().strftime('%Y-%m-%d')}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendation Section
            st.markdown("### 🎯 Course Recommendation")
            
            # Main recommendation card
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
                margin: 1rem 0;
            ">
                <h3 style="color: #FFD700; margin-bottom: 1rem; font-size: 1.5rem;">🏆 Recommended Course</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{course}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Alternative suggestions
            st.markdown("### ✨ Alternative Course Options")
            
            # Create cards for alternative courses
            alt_courses = [c for c in top_courses if c != course]
            if len(alt_courses) >= 2:
                col1, col2 = st.columns(2)
                for i, alt_course in enumerate(alt_courses[:2]):
                    with col1 if i == 0 else col2:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            padding: 1.5rem;
                            border-radius: 10px;
                            border: 2px solid #e9ecef;
                            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                            text-align: center;
                            margin: 0.5rem 0;
                        ">
                            <h5 style="color: #667eea; margin-bottom: 0.5rem;">Option {i+1}</h5>
                            <p style="color: #2c3e50; font-weight: 600; margin: 0;">{alt_course}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                for i, alt_course in enumerate(alt_courses):
                    st.markdown(f"""
                    <div style="
                        background: white;
                        padding: 1rem;
                        border-radius: 8px;
                        border-left: 4px solid #4facfe;
                        margin: 0.5rem 0;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    ">
                        <strong style="color: #4facfe;">Alternative {i+1}:</strong> 
                        <span style="color: #2c3e50;">{alt_course}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # ---- Student Report ----
            student_report = {
                "Interest": interest,
                "Percentage": percentage,
                "Goal": goal,
                "Recommended Course": course,
                "Other Suggestions": ", ".join(top_courses),
                "Generated Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Course Details and Statistics
            st.markdown("### 📊 Course Analysis")
            
            # Get similar students data
            similar_students = data[
                (data['Interest'] == interest.lower()) & 
                (data['Goal'] == goal.lower())
            ]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    label="📚 Similar Students",
                    value=len(similar_students),
                    help="Students with same interest and goal"
                )
            with col2:
                avg_percentage = similar_students['Percentage'].mean() if len(similar_students) > 0 else percentage
                st.metric(
                    label="📊 Avg Percentage",
                    value=f"{avg_percentage:.1f}%",
                    delta=f"{percentage - avg_percentage:.1f}%"
                )
            with col3:
                course_popularity = len(data[data['Recommended_Course'] == course])
                st.metric(
                    label="🏆 Course Popularity",
                    value=f"{course_popularity} students",
                    help="Total students recommended this course"
                )
            with col4:
                confidence = max(probs) * 100
                st.metric(
                    label="🎯 Confidence",
                    value=f"{confidence:.1f}%",
                    help="Model confidence in recommendation"
                )

            # Enhanced Export Options
            st.markdown("### 💾 Export Your Report")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv_buf = io.StringIO()
                pd.DataFrame([student_report]).to_csv(csv_buf, index=False)
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv_buf.getvalue(),
                    file_name=f"student_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # PDF Export
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.colors import HexColor

                    pdf_buf = io.BytesIO()
                    c = canvas.Canvas(pdf_buf, pagesize=letter)
                    
                    # Header
                    c.setFillColor(HexColor('#667eea'))
                    c.rect(0, 720, 612, 72, fill=1)
                    c.setFillColor(HexColor('#FFFFFF'))
                    c.setFont("Helvetica-Bold", 20)
                    c.drawString(50, 750, "AI Student Course Advisor Report")
                    
                    # Content
                    c.setFillColor(HexColor('#000000'))
                    y_position = 680
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(50, y_position, "Student Profile")
                    
                    y_position -= 30
                    c.setFont("Helvetica", 12)
                    for k, v in student_report.items():
                        c.drawString(50, y_position, f"{k}: {v}")
                        y_position -= 20

                    c.showPage()
                    c.save()
                    pdf_buf.seek(0)

                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buf,
                        file_name=f"student_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except ImportError:
                    st.info("📝 PDF export requires `reportlab`. Install with: `pip install reportlab`")

# Analytics Dashboard Tab
with tab2:
    st.markdown("### 📊 Analytics Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Model Accuracy</h3>
            <h2>{:.1f}%</h2>
        </div>
        """.format(acc*100), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Total Students</h3>
            <h2>{}</h2>
        </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col3:
        unique_courses = len(data['Recommended_Course'].unique())
        st.markdown("""
        <div class="metric-card">
            <h3>🏷️ Unique Courses</h3>
            <h2>{}</h2>
        </div>
        """.format(unique_courses), unsafe_allow_html=True)
    
    with col4:
        avg_percentage = data['Percentage'].mean()
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Avg Percentage</h3>
            <h2>{:.1f}%</h2>
        </div>
        """.format(avg_percentage), unsafe_allow_html=True)
    
    # Interactive Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Interest Distribution")
        interest_counts = data['Interest'].value_counts()
        fig_interest = px.pie(
            values=interest_counts.values,
            names=interest_counts.index,
            title="Student Interest Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_interest.update_layout(height=400)
        st.plotly_chart(fig_interest, use_container_width=True)
    
    with col2:
        st.markdown("#### 🚀 Goal Distribution")
        goal_counts = data['Goal'].value_counts()
        fig_goal = px.bar(
            x=goal_counts.index,
            y=goal_counts.values,
            title="Future Goals Distribution",
            color=goal_counts.values,
            color_continuous_scale="viridis"
        )
        fig_goal.update_layout(height=400, xaxis_title="Goals", yaxis_title="Count")
        st.plotly_chart(fig_goal, use_container_width=True)
    
    # Course Popularity Analysis
    st.markdown("#### 🏆 Top Recommended Courses")
    course_counts = data['Recommended_Course'].value_counts().head(10)
    fig_courses = px.bar(
        x=course_counts.values,
        y=course_counts.index,
        orientation='h',
        title="Most Popular Course Recommendations",
        color=course_counts.values,
        color_continuous_scale="blues"
    )
    fig_courses.update_layout(height=500, xaxis_title="Number of Students", yaxis_title="Courses")
    st.plotly_chart(fig_courses, use_container_width=True)
    
    # Percentage Distribution
    st.markdown("#### 📈 Academic Performance Analysis")
    fig_hist = px.histogram(
        data,
        x='Percentage',
        nbins=20,
        title="Distribution of Academic Percentages",
        color_discrete_sequence=['#667eea']
    )
    fig_hist.update_layout(height=400, xaxis_title="Percentage", yaxis_title="Number of Students")
    st.plotly_chart(fig_hist, use_container_width=True)

# Model Performance Tab
with tab3:
    st.markdown("### 📈 Model Performance Analysis")
    
    # Performance Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        
        # Create interactive confusion matrix with Plotly
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=le_course.classes_,
            y=le_course.classes_,
            colorscale='Blues',
            showscale=True
        ))
        fig_cm.update_layout(
            title="Confusion Matrix Heatmap",
            xaxis_title="Predicted Course",
            yaxis_title="Actual Course",
            height=500
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("#### 📉 Feature Importance")
        feature_importance = model.feature_importances_
        features = ['Interest', 'Percentage', 'Goal']
        
        fig_importance = px.bar(
            x=features,
            y=feature_importance,
            title="Feature Importance in Course Recommendation",
            color=feature_importance,
            color_continuous_scale="viridis"
        )
        fig_importance.update_layout(height=500, xaxis_title="Features", yaxis_title="Importance")
        st.plotly_chart(fig_importance, use_container_width=True)

    # Classification Report
    st.markdown("#### 📊 Detailed Classification Report")
    report = classification_report(y, y_pred, target_names=le_course.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Style the dataframe
    styled_df = report_df.style.background_gradient(cmap="Blues").format("{:.3f}")
    st.dataframe(styled_df, use_container_width=True)
    
    # Model Insights
    st.markdown("#### 💡 Model Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        precision_avg = report_df.loc['macro avg', 'precision']
        st.metric("Macro Avg Precision", f"{precision_avg:.3f}")
    
    with col2:
        recall_avg = report_df.loc['macro avg', 'recall']
        st.metric("Macro Avg Recall", f"{recall_avg:.3f}")
    
    with col3:
        f1_avg = report_df.loc['macro avg', 'f1-score']
        st.metric("Macro Avg F1-Score", f"{f1_avg:.3f}")

    # Export Model Performance
    st.markdown("#### 💾 Export Model Performance")
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer)
    st.download_button(
        label="📊 Download Classification Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Dataset Explorer Tab
with tab4:
    st.markdown("### 🗂️ Dataset Explorer")
    
    # Dataset Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())
    
    # Interactive Data Table
    st.markdown("#### 📋 Interactive Data Table")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        interest_filter = st.multiselect("Filter by Interest", data['Interest'].unique(), default=data['Interest'].unique())
    with col2:
        goal_filter = st.multiselect("Filter by Goal", data['Goal'].unique(), default=data['Goal'].unique())
    with col3:
        percentage_range = st.slider("Percentage Range", int(data['Percentage'].min()), int(data['Percentage'].max()), (int(data['Percentage'].min()), int(data['Percentage'].max())))
    
    # Apply filters
    filtered_data = data[
        (data['Interest'].isin(interest_filter)) &
        (data['Goal'].isin(goal_filter)) &
        (data['Percentage'] >= percentage_range[0]) &
        (data['Percentage'] <= percentage_range[1])
    ]
    
    st.dataframe(
        filtered_data,
        use_container_width=True,
        height=400
    )
    
    # Dataset Statistics
    st.markdown("#### 📊 Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numerical Statistics**")
        st.dataframe(data.describe(), use_container_width=True)
    
    with col2:
        st.markdown("**Categorical Statistics**")
        categorical_stats = pd.DataFrame({
            'Interest': data['Interest'].value_counts(),
            'Goal': data['Goal'].value_counts()
        }).fillna(0)
        st.dataframe(categorical_stats, use_container_width=True)
    
    # Export Dataset
    st.markdown("#### 💾 Export Dataset")
    csv_data = io.StringIO()
    filtered_data.to_csv(csv_data, index=False)
    st.download_button(
        label="📊 Download Filtered Dataset (CSV)",
        data=csv_data.getvalue(),
        file_name=f"filtered_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 2rem -2rem -2rem -2rem; border-radius: 20px 20px 0 0;">
    <p style="color: white; font-size: 1.1rem; margin: 0;">🚀 Built with Streamlit & Machine Learning | AI Student Course Advisor</p>
    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.5rem 0 0 0;">Empowering students with intelligent course recommendations</p>
</div>
""", unsafe_allow_html=True)
