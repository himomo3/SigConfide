import streamlit as st
import streamlit.components.v1 as components
import os

# Run with:
# python -m streamlit run view_plots.py

st.set_page_config(layout="wide", page_title="Plot Viewer")

patients = ["PD24196", "PD8609", "PD13608"]
plot_types = [
    ("Element bounds", "element_bounds.png"),
    ("Error density", "error_density.png"),
    ("Spatial PCA", "spatial_pca.png")
]
output_dir = "comparison_output"

data_sources = {}
if os.path.exists(output_dir):
    if any(os.path.isdir(os.path.join(output_dir, pt)) for pt in patients):
        data_sources["Legacy (Root)"] = output_dir
    
    for d in os.listdir(output_dir):
        if d in patients:
            continue
        full_path = os.path.join(output_dir, d)
        if os.path.isdir(full_path):
            if any(os.path.isdir(os.path.join(full_path, pt)) for pt in patients):
                data_sources[d] = full_path
            elif d == "synthetic2700_all":
                data_sources[d] = full_path

if not data_sources:
    data_sources["Legacy (Root)"] = output_dir

st.sidebar.title("Data Source")
selected_source = st.sidebar.selectbox("Choose data to view", list(data_sources.keys()))
base_dir = data_sources[selected_source]

# Define columns: 1 for the row label, and 1 for each patient
col_ratios = [1.5, 3, 3, 3]

# ---- Single Plot Navigation Mode ----
if "pt_idx" not in st.session_state:
    st.session_state.pt_idx = 0
if "plot_idx" not in st.session_state:
    st.session_state.plot_idx = 0

def prev_pt(): st.session_state.pt_idx = (st.session_state.pt_idx - 1) % len(patients)
def next_pt(): st.session_state.pt_idx = (st.session_state.pt_idx + 1) % len(patients)
def prev_plot(): st.session_state.plot_idx = (st.session_state.plot_idx - 1) % len(plot_types)
def next_plot(): st.session_state.plot_idx = (st.session_state.plot_idx + 1) % len(plot_types)

has_patients = any(os.path.isdir(os.path.join(base_dir, pt)) for pt in patients)

st.sidebar.title("View Modes")
single_plot_mode = st.sidebar.checkbox("Enable Single Plot View", value=True, disabled=not has_patients)

if not has_patients:
    st.markdown("<h2 style='text-align: center;'>Global Metrics Only (Synthetic Benchmark)</h2>", unsafe_allow_html=True)
    st.markdown("---")

elif single_plot_mode:
    st.markdown("""
        <style>
        div[data-testid="stImage"] img {
            max-height: 75vh !important;
            object-fit: contain !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center;'>Single Plot View</h2>", unsafe_allow_html=True)
    
    # Navigation controls (hidden by JS)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("PrevPtBtn", on_click=prev_pt)
    with col2:
        st.button("NextPtBtn", on_click=next_pt)
    with col3:
        st.button("PrevPlotBtn", on_click=prev_plot)
    with col4:
        st.button("NextPlotBtn", on_click=next_plot)

    # The actual plot viewer
    curr_pt = patients[st.session_state.pt_idx]
    curr_plot_name, curr_file_name = plot_types[st.session_state.plot_idx]
    
    st.markdown(f"<h3 style='text-align: center;'>{curr_pt} - {curr_plot_name}</h3>", unsafe_allow_html=True)
    
    img_path = os.path.join(base_dir, curr_pt, curr_file_name)
    if os.path.exists(img_path):
        st.image(img_path, width="stretch")
    else:
        st.error(f"Image not found: {img_path}")

    # JavaScript to handle arrow keys and hide buttons
    components.html(
        """
        <script>
        const doc = window.parent.document;
        
        if (!doc.getElementById('arrow-key-script')) {
            const script = doc.createElement('script');
            script.id = 'arrow-key-script';
            script.innerHTML = `
                document.addEventListener('keydown', function(e) {
                    const isSinglePlot = document.querySelector('h2') && document.querySelector('h2').innerText === 'Single Plot View';
                    if (!isSinglePlot) return;
                    
                    if(["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].indexOf(e.code) > -1) {
                        e.preventDefault();
                    }
                    const btns = Array.from(document.querySelectorAll('button'));
                    if (e.key === 'ArrowLeft') {
                        const btn = btns.find(el => el.innerText === 'PrevPtBtn');
                        if (btn) btn.click();
                    } else if (e.key === 'ArrowRight') {
                        const btn = btns.find(el => el.innerText === 'NextPtBtn');
                        if (btn) btn.click();
                    } else if (e.key === 'ArrowUp') {
                        const btn = btns.find(el => el.innerText === 'PrevPlotBtn');
                        if (btn) btn.click();
                    } else if (e.key === 'ArrowDown') {
                        const btn = btns.find(el => el.innerText === 'NextPlotBtn');
                        if (btn) btn.click();
                    }
                });
            `;
            doc.body.appendChild(script);
        }

        // Hide the buttons immediately
        const buttons = Array.from(doc.querySelectorAll('button'));
        buttons.forEach(b => {
            if (b.innerText.includes('PtBtn') || b.innerText.includes('PlotBtn')) {
                const col = b.closest('div[data-testid="stColumn"]');
                if (col) col.style.display = 'none';
            }
        });
        </script>
        """,
        height=0
    )
    
    st.markdown("---")

elif has_patients:
    # Cleanup previously hidden inline styles from JS
    components.html("""
        <script>
        const doc = window.parent.document;
        const cols = Array.from(doc.querySelectorAll('div[data-testid="stColumn"]'));
        cols.forEach(col => { col.style.display = ''; });
        </script>
    """, height=0)

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
                cols[i+1].image(img_path, width="stretch")
            else:
                cols[i+1].error(f"Image not found: {img_path}")

st.markdown("---")
st.markdown("*Comparison of SFS Geometry vs Bootstrap Statistics*")

st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Signature Modifications Timeline</h3>", unsafe_allow_html=True)
if has_patients:
    for pt in patients:
        img_path1 = os.path.join(base_dir, pt, "P_cosine_similarity.png")
        img_path2 = os.path.join(base_dir, pt, "P_cosine_similarity_per_sig.png")
        img_path3 = os.path.join(base_dir, pt, "P_cosine_similarity_dist.png")
        if os.path.exists(img_path1) and os.path.exists(img_path2):
            cols = st.columns(2)
            cols[0].image(img_path1, use_container_width=True)
            cols[1].image(img_path2, use_container_width=True)
            if os.path.exists(img_path3):
                cols_dist = st.columns([1, 2, 1])
                cols_dist[1].image(img_path3, use_container_width=True)
            break
        elif os.path.exists(img_path1):
            cols = st.columns([1, 2, 1])
            cols[1].image(img_path1, use_container_width=True)
            if os.path.exists(img_path3):
                cols_dist = st.columns([1, 2, 1])
                cols_dist[1].image(img_path3, use_container_width=True)
            break
else:
    img_path1 = os.path.join(base_dir, "P_cosine_similarity.png")
    img_path2 = os.path.join(base_dir, "P_cosine_similarity_per_sig.png")
    img_path3 = os.path.join(base_dir, "P_cosine_similarity_dist.png")
    if os.path.exists(img_path1) and os.path.exists(img_path2):
        cols = st.columns(2)
        cols[0].image(img_path1, use_container_width=True)
        cols[1].image(img_path2, use_container_width=True)
        if os.path.exists(img_path3):
            cols_dist = st.columns([1, 2, 1])
            cols_dist[1].image(img_path3, use_container_width=True)
    elif os.path.exists(img_path1):
        cols = st.columns([1, 2, 1])
        cols[1].image(img_path1, use_container_width=True)
        if os.path.exists(img_path3):
            cols_dist = st.columns([1, 2, 1])
            cols_dist[1].image(img_path3, use_container_width=True)

st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Exposure Differences (E_other - E_opt)</h3>", unsafe_allow_html=True)
img_path_global_box = os.path.join(base_dir, "global_exposure_diff_box.png")
if os.path.exists(img_path_global_box):
    cols = st.columns([1, 4, 1])
    cols[1].image(img_path_global_box, use_container_width=True)

st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Difference in Pairwise Cosine Similarity</h3>", unsafe_allow_html=True)
if has_patients:
    for pt in patients:
        img_path_heatmap = os.path.join(base_dir, pt, "pairwise_cosine_similarity.png")
        if os.path.exists(img_path_heatmap):
            cols = st.columns([1, 3, 1])
            cols[1].image(img_path_heatmap, use_container_width=True)
            break
else:
    img_path_heatmap = os.path.join(base_dir, "pairwise_cosine_similarity.png")
    if os.path.exists(img_path_heatmap):
        cols = st.columns([1, 3, 1])
        cols[1].image(img_path_heatmap, use_container_width=True)

st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Global Reconstruction Errors Density</h3>", unsafe_allow_html=True)
img_path_iter1 = os.path.join(base_dir, "global_recon_err_density_iter1.png")
img_path_iter1000 = os.path.join(base_dir, "global_recon_err_density_iter1000.png")
img_path_qp = os.path.join(base_dir, "global_recon_err_density_qp.png")

if os.path.exists(img_path_iter1) and os.path.exists(img_path_iter1000):
    cols = st.columns(2)
    cols[0].image(img_path_iter1, width="stretch", caption="After 1 Iteration")
    cols[1].image(img_path_iter1000, width="stretch", caption="After 1000 Iterations")

if os.path.exists(img_path_qp):
    cols_qp = st.columns([1, 2, 1])
    cols_qp[1].image(img_path_qp, width="stretch", caption="Original QP")
