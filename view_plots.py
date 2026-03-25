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
base_dir = "comparison_output"

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

st.sidebar.title("View Modes")
single_plot_mode = st.sidebar.checkbox("Enable Single Plot View", value=True)

if single_plot_mode:
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

else:
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
