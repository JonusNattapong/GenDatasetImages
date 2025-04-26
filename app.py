"""Streamlit web interface for dataset generation and management."""
import streamlit as st
import os
import sys
import yaml
import time
from datetime import datetime
import threading
from queue import Queue
import traceback

# --- Add project root to sys.path ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Import backend modules ---
try:
    from src.config_loader import load_config
    from src.a1111_client import A1111Client
    from src.generator import generate_parameter_combinations
    from src.dataset_builder import DatasetBuilder
    from src.dataset_manager import DatasetManager
except ImportError as e:
    st.error(f"Error importing backend modules: {e}. Ensure 'src' directory is accessible.")
    st.stop()

# --- App State Initialization ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0.0
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'generated_images_data' not in st.session_state:
    st.session_state.generated_images_data = []
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Idle."
if 'total_combinations' not in st.session_state:
    st.session_state.total_combinations = 0
if 'generation_thread' not in st.session_state:
    st.session_state.generation_thread = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'progress_queue' not in st.session_state:
    st.session_state.progress_queue = Queue()

# --- Worker Function ---
def generation_worker(api_url_w, dataset_name_w, output_base_dir_w,
                     base_params_w, variations_w, queue):
    """Background worker for image generation process."""
    try:
        queue.put({"log": "Worker thread started..."})

        try:
            client = A1111Client(api_url_w)
            builder = DatasetBuilder(dataset_name_w, output_base_dir_w)
            queue.put({"log": "Backend initialized."})
        except Exception as e:
            raise RuntimeError(f"Failed to initialize backend: {e}")

        queue.put({"status": "Calculating parameter combinations..."})
        param_combinations = list(generate_parameter_combinations(base_params_w, variations_w))
        total_combinations = len(param_combinations)
        st.session_state.total_combinations = total_combinations
        queue.put({"status": f"Found {total_combinations} combinations."})

        if total_combinations == 0:
            raise ValueError("No parameter combinations generated. Check variations.")

        generated_count = 0
        failed_count = 0

        for i, params in enumerate(param_combinations):
            current_job_text = f"Generating image {i+1}/{total_combinations}..."
            progress = (i + 1) / total_combinations
            queue.put({
                "status": current_job_text,
                "progress": progress,
                "log": f"INFO: {current_job_text}\nDEBUG: {params}"
            })

            try:
                image_bytes, generation_info = client.generate_image(params)

                if image_bytes and generation_info:
                    metadata = generation_info.copy()
                    metadata['request_parameters'] = params
                    builder.add_image(image_bytes, metadata)
                    generated_count += 1
                    caption = f"Image {i+1}: {metadata.get('filename', 'N/A')}"
                    queue.put({
                        "log": f"SUCCESS: Image {i+1} saved.",
                        "image": (caption, image_bytes)
                    })
                else:
                    failed_count += 1
                    queue.put({"log": f"ERROR: Image {i+1} generation failed."})

            except Exception as e:
                failed_count += 1
                error_trace = traceback.format_exc()
                queue.put({"log": f"CRITICAL: Error on image {i+1}: {e}\n{error_trace}"})

        final_status = f"Finished. Generated: {generated_count}, Failed: {failed_count}."
        metadata_path = builder.finalize_dataset()
        if metadata_path:
            final_status += f"\nDataset saved. Metadata: {metadata_path}"

        queue.put({"status": final_status, "progress": 1.0, "done": True})

    except Exception as e:
        error_trace = traceback.format_exc()
        queue.put({"status": f"Worker Error: {e}", "error": str(e), 
                  "log": error_trace, "done": True})

# --- App Configuration ---
st.set_page_config(page_title="GenDatasetImages", page_icon="🖼️", layout="wide")

# --- Navigation ---
page = st.sidebar.radio(
    "Navigation",
    ["Generate", "Manage Datasets"],
    format_func=lambda x: "🖼️ " + x if x == "Generate" else "📁 " + x
)

# --- Generation Page ---
if page == "Generate":
    st.title("🖼️ GenDatasetImages: Dataset Generation")
    st.markdown("Configure and generate image datasets using the Automatic1111 API.")

    # --- Configuration (Sidebar) ---
    st.sidebar.header("Generation Settings")
    
    api_url = st.sidebar.text_input("A1111 API URL", 
                                   value="http://127.0.0.1:7860",
                                   disabled=st.session_state.is_running)
    dataset_name = st.sidebar.text_input("Dataset Name",
                                       value=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M')}",
                                       disabled=st.session_state.is_running)
    output_base_dir = st.sidebar.text_input("Output Directory",
                                          value="./output_datasets",
                                          disabled=st.session_state.is_running)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Base Parameters")
    base_prompt = st.sidebar.text_area("Base Prompt",
                                     "a photograph of an astronaut riding a horse",
                                     height=100,
                                     disabled=st.session_state.is_running)
    base_neg_prompt = st.sidebar.text_area("Base Negative Prompt",
                                         "blurry, low quality, text, watermark",
                                         height=100,
                                         disabled=st.session_state.is_running)
    base_steps = st.sidebar.number_input("Steps", 1, 150, 25,
                                       disabled=st.session_state.is_running)
    base_cfg = st.sidebar.number_input("CFG Scale", 1.0, 30.0, 7.0, 0.5,
                                     disabled=st.session_state.is_running)
    base_width = st.sidebar.number_input("Width", 64, 2048, 512, 64,
                                       disabled=st.session_state.is_running)
    base_height = st.sidebar.number_input("Height", 64, 2048, 512, 64,
                                        disabled=st.session_state.is_running)
    base_seed = st.sidebar.number_input("Seed (-1 for random)", -1, None, -1,
                                      disabled=st.session_state.is_running)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameter Variations")
    prompt_variations_str = st.sidebar.text_area(
        "Prompt Variations (one per line)",
        "a painting of a cat wearing a hat\na sketch of a dog playing fetch",
        disabled=st.session_state.is_running
    )
    seed_variations_str = st.sidebar.text_area(
        "Seed Variations (one per line)",
        "12345\n54321\n98765",
        disabled=st.session_state.is_running
    )

    # --- Generation Control ---
    st.header("Generation Control")
    start_button = st.button("🚀 Start Generation",
                            disabled=st.session_state.is_running)

    # --- Progress Display ---
    st.markdown("---")
    st.header("Progress & Results")

    status_placeholder = st.empty()
    status_placeholder.info(st.session_state.status_message)
    progress_bar = st.progress(st.session_state.progress)

    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    with st.expander("Logs", expanded=False):
        st.code('\n'.join(map(str, st.session_state.log_messages[-15:])))

    if st.session_state.generated_images_data:
        st.subheader("Generated Images Preview")
        cols = st.columns(4)
        for idx, (caption, img_bytes) in enumerate(st.session_state.generated_images_data):
            try:
                cols[idx % 4].image(img_bytes, caption=caption, width=200)
            except Exception as e:
                cols[idx % 4].warning(f"Could not display image {idx+1}: {e}")

# --- Dataset Management Page ---
else:
    st.title("📁 Dataset Management")
    st.markdown("Manage, archive, and export your generated datasets.")
    
    manager = DatasetManager("./output_datasets")
    datasets = manager.list_datasets(include_archived=True)
    
    if not datasets:
        st.info("No datasets found. Generate some images first!")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Available Datasets")
            selected_dataset = st.selectbox(
                "Select a dataset",
                options=list(datasets.keys()),
                format_func=lambda x: f"{x} ({'🗄️ Archived' if datasets[x]['status'] == 'archived' else '📊 Active'})"
            )
        
        if selected_dataset:
            with col2:
                st.subheader("Dataset Details")
                info = manager.get_dataset_info(selected_dataset.split('_')[0])
                
                st.write(f"**Status:** {info['status'].title()}")
                st.write(f"**Created:** {info['created']}")
                st.write(f"**Images:** {info['image_count']}")
                st.write(f"**Size:** {info['total_size'] / 1024 / 1024:.2f} MB")
                
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    if info['status'] == 'active':
                        if st.button("📦 Archive"):
                            try:
                                manager.archive_dataset(selected_dataset.split('_')[0])
                                st.success("Dataset archived!")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error archiving dataset: {e}")
                
                with col2b:
                    if st.button("⬇️ Export ZIP"):
                        try:
                            with st.spinner("Creating ZIP file..."):
                                zip_path = manager.export_dataset(selected_dataset.split('_')[0])
                            with open(zip_path, "rb") as f:
                                st.download_button(
                                    "📥 Download ZIP",
                                    f,
                                    file_name=f"{selected_dataset}.zip",
                                    mime="application/zip"
                                )
                        except Exception as e:
                            st.error(f"Error exporting dataset: {e}")
                
                with col2c:
                    if st.button("🗑️ Delete", type="secondary"):
                        if st.checkbox("Confirm deletion?"):
                            try:
                                manager.delete_dataset(selected_dataset.split('_')[0])
                                st.success("Dataset deleted!")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error deleting dataset: {e}")
                
                if info.get('metadata_sample'):
                    with st.expander("Sample Metadata"):
                        st.json(info['metadata_sample'][0])

# --- Background Task Handling ---
if st.session_state.is_running or not st.session_state.progress_queue.empty():
    while not st.session_state.progress_queue.empty():
        msg = st.session_state.progress_queue.get()
        if "log" in msg:
            st.session_state.log_messages.append(msg["log"])
        if "status" in msg:
            st.session_state.status_message = msg["status"]
        if "progress" in msg:
            st.session_state.progress = msg["progress"]
        if "image" in msg:
            st.session_state.generated_images_data.append(msg["image"])
        if "error" in msg:
            st.session_state.error_message = msg["error"]
            st.session_state.is_running = False
        if msg.get("done", False):
            st.session_state.is_running = False
            st.session_state.status_message = msg.get("status", "Finished.")
            st.session_state.progress = msg.get("progress", 1.0)

    status_placeholder.info(st.session_state.status_message)
    progress_bar.progress(st.session_state.progress)
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    if (st.session_state.is_running and st.session_state.generation_thread 
            and not st.session_state.generation_thread.is_alive()):
        st.session_state.is_running = False
        st.session_state.error_message = (st.session_state.error_message or 
                                        "Thread stopped unexpectedly.")
        st.session_state.status_message = "Error: Thread stopped."
        st.error(st.session_state.error_message)

    if st.session_state.is_running:
        time.sleep(0.5)
        st.experimental_rerun()

# --- Start Generation Logic ---
if start_button and not st.session_state.is_running:
    st.session_state.is_running = True
    st.session_state.progress = 0.0
    st.session_state.log_messages = ["Initiating..."]
    st.session_state.generated_images_data = []
    st.session_state.status_message = "Parsing inputs..."
    st.session_state.error_message = None
    st.session_state.total_combinations = 0

    while not st.session_state.progress_queue.empty():
        try:
            st.session_state.progress_queue.get_nowait()
        except Exception:
            break

    try:
        prompt_variations = [p.strip() for p in prompt_variations_str.split('\n') if p.strip()]
        try:
            raw_seeds = [s.strip() for s in seed_variations_str.split('\n') if s.strip()]
            seed_variations = [int(s) for s in raw_seeds]
            if len(raw_seeds) != len(seed_variations):
                raise ValueError("Non-integer value found in Seed Variations.")
        except ValueError as e:
            raise ValueError(f"Invalid seeds: {e}. Enter one integer per line.")

        base_params = {
            "prompt": base_prompt,
            "negative_prompt": base_neg_prompt,
            "steps": base_steps,
            "cfg_scale": base_cfg,
            "width": base_width,
            "height": base_height,
            "seed": base_seed
        }

        variations = {}
        if prompt_variations:
            variations['prompt'] = prompt_variations
        if seed_variations:
            variations['seed'] = seed_variations

        if not variations and not base_params.get('prompt'):
            raise ValueError("Please provide at least a base prompt or variations.")

        st.session_state.generation_thread = threading.Thread(
            target=generation_worker,
            args=(api_url, dataset_name, output_base_dir,
                  base_params, variations,
                  st.session_state.progress_queue),
            daemon=True
        )
        st.session_state.generation_thread.start()
        st.experimental_rerun()

    except Exception as e:
        st.session_state.is_running = False
        st.session_state.error_message = str(e)
        st.error(str(e))
