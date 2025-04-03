import pandas as pd
import plotly.express as px
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("hackathon_participants.csv")

# Apply Custom Styling
st.markdown(
    """
    <style>
    /* Background color */
    .stApp { background-color: #eef1f7; }
    .css-1aumxhk { background-color: #f5f5f5 !important; }

    /* Text color adjustments */
    h1, h2, h3, h4, h5, h6, p, label { color: black !important; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hackathon Event Analysis Dashboard")

# Sidebar Filters
domains = df["Domain"].unique().tolist()
states = df["State"].unique().tolist()
colleges = df["College"].unique().tolist()

domain_filter = st.sidebar.multiselect("Select Domains", domains, default=domains)
state_filter = st.sidebar.multiselect("Select States", states, default=states)
college_filter = st.sidebar.multiselect("Select Colleges", colleges, default=colleges)

# Date Range Selector
df["Date"] = pd.to_datetime(df["Date"])
date_range = st.sidebar.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])

# Search Boxes
search_name = st.sidebar.text_input("Search Participant Name")
search_college = st.sidebar.text_input("Search College Name")

# Filtered Data
df_filtered = df[
    (df["Domain"].isin(domain_filter)) &
    (df["State"].isin(state_filter)) &
    (df["College"].isin(college_filter)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

if search_name:
    df_filtered = df_filtered[df_filtered["Name"].str.contains(search_name, case=False, na=False)]
if search_college:
    df_filtered = df_filtered[df_filtered["College"].str.contains(search_college, case=False, na=False)]

st.write("## Filtered Participation Data")
st.dataframe(df_filtered)

# Tabs for Different Sections
tab1, tab2, tab3 = st.tabs(["Participation Trends", "Feedback Analysis", "Image Processing"])

with tab1:
    st.write("## Participation Trends")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_filtered, x="Domain", title="Domain-wise Participation", animation_frame="Day")
        st.plotly_chart(fig1)
    with col2:
        day_wise_participation = df_filtered.groupby("Day").size().reset_index(name="Count")
        fig2 = px.line(day_wise_participation, x="Day", y="Count", title="Day-wise Participation")
        st.plotly_chart(fig2)
    
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(df_filtered, x="College", title="College-wise Participation", size_max=10)
        st.plotly_chart(fig3)
    with col4:
        fig4 = px.bar(df_filtered, x="State", title="State-wise Participation", color="State")
        st.plotly_chart(fig4)
    
    st.write("## Participation Distribution")
    col5, col6 = st.columns(2)
    with col5:
        fig6 = px.pie(df_filtered, names="Domain", title="Domain-wise Distribution", hole=0.4)
        st.plotly_chart(fig6)
    with col6:
        fig7 = px.pie(df_filtered, names="State", title="State-wise Distribution", hole=0.4)
        st.plotly_chart(fig7)

with tab2:
    st.write("## Participant Feedback Analysis")
    for domain in df_filtered["Domain"].unique():
        domain_feedback = df_filtered[df_filtered["Domain"] == domain]["Feedback"].dropna().str.cat(sep=" ")
        if domain_feedback:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(domain_feedback)
            
            st.write(f"### Word Cloud for {domain}")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
            st.write(f"#### Sample Feedback for {domain}")
            st.write(domain_feedback[:500] + "...")

with tab3:
    st.write("## Real-Time Camera & Image Processing")

    # Real-time Camera Capture
    camera_image = st.camera_input("Capture an image using your camera")
    
    # Upload Image
    uploaded_image = st.file_uploader("Or Upload an Image", type=["jpg", "png", "jpeg"])

    # Select Image Source
    image_source = camera_image if camera_image else uploaded_image

    if image_source:
        image = Image.open(image_source)
        st.image(image, caption="Original Image", use_column_width=True)

        # Convert to OpenCV format
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Blur Level Slider
        blur_level = st.slider("Select Blur Level", 1, 25, 5, step=2)

        # Grayscale Conversion
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        st.write("### Grayscale Image")
        st.image(gray_image, caption="Grayscale", use_column_width=True, channels="GRAY")

        # Edge Detection
        edges = cv2.Canny(gray_image, 100, 200)
        st.write("### Edge Detection")
        st.image(edges, caption="Edge Detection", use_column_width=True, channels="GRAY")

        # Gaussian Blur
        gaussian_blur = cv2.GaussianBlur(image_cv, (blur_level, blur_level), 0)
        st.write("### Gaussian Blur")
        st.image(gaussian_blur, caption="Gaussian Blur", use_column_width=True, channels="BGR")

        # Median Blur
        median_blur = cv2.medianBlur(image_cv, blur_level)
        st.write("### Median Blur")
        st.image(median_blur, caption="Median Blur", use_column_width=True, channels="BGR")

        # Black & White (Thresholding)
        _, black_white = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        st.write("### Black & White Image")
        st.image(black_white, caption="Black & White", use_column_width=True, channels="GRAY")

        # RGB Filter
        rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        st.write("### RGB Filtered Image")
        st.image(rgb_image, caption="RGB Filter", use_column_width=True, channels="RGB")
