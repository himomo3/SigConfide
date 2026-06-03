import streamlit as st
import streamlit.components.v1 as components
import os

# Run with:
# python -m streamlit run view_plots.py

st.set_page_config(layout="wide", page_title="Plot Viewer")

# Hardcoded patients is no longer strictly necessary but kept for legacy root compatibility
patients = [
    "PD24196", "PD8609", "PD13608",
    "SP.Syn.Kidney-RCC..S.222",
    "SP.Syn.Stomach-AdenoCA..S.178",
    "SP.Syn.Stomach-AdenoCA..S.300",
    "SP.Syn.Cervix-AdenoCA..S.272",
    "SP.Syn.Breast-AdenoCA..S.177",
    "SP.Syn.Stomach-AdenoCA..S.243",
    "SP.Syn.Bone-Osteosarc..S.80",
    "SP.Syn.Bone-Osteosarc..S.130",
    "SP.Syn.Bone-Osteosarc..S.116"
]
plot_types = [
    ("Element bounds", "element_bounds.png"),
    ("Error density", "error_density.png"),
    ("Spatial PCA", "spatial_pca.png")
]
output_dir = "comparison_output"

data_sources = {}
if os.path.exists(output_dir):
    for d in os.listdir(output_dir):
        full_path = os.path.join(output_dir, d)
        if os.path.isdir(full_path):
            if "synthetic2700_all" in os.listdir(full_path):
                data_sources[d] = os.path.join(full_path, "synthetic2700_all")
            elif "global_computed_data.npz" in os.listdir(full_path):
                data_sources[d] = full_path
            elif any(sub_d.startswith("PD") or sub_d.startswith("SP.Syn") or sub_d.startswith("Patient") for sub_d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, sub_d))):
                data_sources[d] = full_path
                
    if any(d.startswith("PD") or d.startswith("SP.Syn") for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))):
        data_sources["Legacy (Root)"] = output_dir

if not data_sources:
    data_sources["Legacy (Root)"] = output_dir

st.sidebar.title("Data Source")
selected_source = st.sidebar.selectbox("Choose data to view", list(data_sources.keys()))
base_dir = data_sources[selected_source]

# Determine which patients exist in the selected directory
active_patients = []
if os.path.exists(base_dir):
    for pt in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, pt)) and (pt.startswith("PD") or pt.startswith("SP.Syn") or pt.startswith("Patient")):
            active_patients.append(pt)


# Define columns: 1 for the row label, and 1 for each active patient
col_ratios = [1.5] + [3] * len(active_patients)

# ---- Single Plot Navigation Mode ----
if "pt_idx" not in st.session_state:
    st.session_state.pt_idx = 0
if "plot_idx" not in st.session_state:
    st.session_state.plot_idx = 0

def prev_pt(): st.session_state.pt_idx = (st.session_state.pt_idx - 1) % max(1, len(active_patients))
def next_pt(): st.session_state.pt_idx = (st.session_state.pt_idx + 1) % max(1, len(active_patients))
def prev_plot(): st.session_state.plot_idx = (st.session_state.plot_idx - 1) % len(plot_types)
def next_plot(): st.session_state.plot_idx = (st.session_state.plot_idx + 1) % len(plot_types)

has_patients = len(active_patients) > 0

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
    if len(active_patients) > 0:
        if st.session_state.pt_idx >= len(active_patients):
            st.session_state.pt_idx = 0
        curr_pt = active_patients[st.session_state.pt_idx]
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
    for i, pt in enumerate(active_patients):
        headers[i+1].markdown(f"<h3 style='text-align: center;'>{pt}</h3>", unsafe_allow_html=True)

    # Plot Rows
    for plot_name, file_name in plot_types:
        cols = st.columns(col_ratios, vertical_alignment="center")
        
        # Row Label
        cols[0].markdown(f"<h4 style='text-align: center; margin-top: 50%;'>{plot_name}</h4>", unsafe_allow_html=True)
        
        # Images per patient
        for i, pt in enumerate(active_patients):
            img_path = os.path.join(base_dir, pt, file_name)
            if os.path.exists(img_path):
                cols[i+1].image(img_path, width="stretch")
            else:
                cols[i+1].error(f"Image not found: {img_path}")

st.markdown("---")
st.markdown("*Comparison of SFS Geometry vs Bootstrap Statistics*")

import pandas as pd
metrics_file = os.path.join(base_dir, "metrics_summary.csv")
if os.path.exists(metrics_file):
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>Prediction Accuracy Metrics</h3>", unsafe_allow_html=True)
    df_metrics = pd.read_csv(metrics_file)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Signature Modifications Timeline</h3>", unsafe_allow_html=True)
found_timeline = False
if has_patients:
    for pt in active_patients:
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
            found_timeline = True
            break
        elif os.path.exists(img_path1):
            cols = st.columns([1, 2, 1])
            cols[1].image(img_path1, use_container_width=True)
            if os.path.exists(img_path3):
                cols_dist = st.columns([1, 2, 1])
                cols_dist[1].image(img_path3, use_container_width=True)
            found_timeline = True
            break

if not found_timeline:
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
st.markdown("<h3 style='text-align: center;'>Pairwise Signature Cosine Similarities</h3>", unsafe_allow_html=True)
found_heatmap = False

def _try_show_heatmaps(search_dir):
    """Try to display heatmaps from search_dir. Returns True if found."""
    orig_path = os.path.join(search_dir, "pairwise_cosine_similarity_original_final.png")
    diff_path = os.path.join(search_dir, "pairwise_cosine_similarity_diff.png")
    legacy_path = os.path.join(search_dir, "pairwise_cosine_similarity.png")
    
    if os.path.exists(orig_path):
        st.image(orig_path, use_container_width=True)
        if os.path.exists(diff_path):
            cols = st.columns([1, 3, 1])
            cols[1].image(diff_path, use_container_width=True)
        return True
    elif os.path.exists(legacy_path):
        cols = st.columns([1, 3, 1])
        cols[1].image(legacy_path, use_container_width=True)
        return True
    return False

if has_patients:
    for pt in active_patients:
        if _try_show_heatmaps(os.path.join(base_dir, pt)):
            found_heatmap = True
            break

if not found_heatmap:
    _try_show_heatmaps(base_dir)

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
