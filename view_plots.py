import streamlit as st
import os

# Run with:
# python -m streamlit run view_plots.py

st.set_page_config(layout="wide", page_title="Plot Viewer")

patients = ["PD8609", "PD13608", "PD24196"]
plot_types = [
    ("Element bounds", "element_bounds.png"),
    ("Error density", "error_density.png"),
    ("Spatial PCA", "spatial_pca.png")
]
base_dir = "comparison_output"

# Define columns: 1 for the row label, and 1 for each patient
col_ratios = [1.5, 3, 3, 3]

# Header Row
headers = st.columns(col_ratios)
headers[0].write("") # Empty corner
for i, pt in enumerate(patients):
    headers[i+1].markdown(f"<h3 style='text-align: center;'>{pt}</h3>", unsafe_allow_html=True)

# Plot Rows
for plot_name, file_name in plot_types:
    cols = st.columns(col_ratios, vertical_alignment="center")
    
    # Row Label
    cols[0].markdown(f"<h4 style='text-align: center; margin-top: 50%;'>{plot_name}</h4>", unsafe_allow_html=True)
    
    # Images per patient
    for i, pt in enumerate(patients):
        img_path = os.path.join(base_dir, pt, file_name)
        if os.path.exists(img_path):
            cols[i+1].image(img_path, use_container_width=True)
        else:
            cols[i+1].error(f"Image not found: {img_path}")

st.markdown("---")
st.markdown("*Comparison of SFS Geometry vs Bootstrap Statistics*")
