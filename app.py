import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas

import logic

st.set_page_config(page_title="StudioAI", layout="wide", page_icon="AI")


@st.cache_resource
def get_models():
    return logic.load_models_cached()


try:
    pipe_txt2img, pipe_inpaint = get_models()
except Exception as exc:
    st.error(f"Error loading models: {exc}")
    st.stop()

st.title("StudioAI: Stable Diffusion Image Suite")

with st.sidebar:
    st.header("Parameters")
    steps = st.slider("Quality Steps", 15, 50, 40)
    cfg = st.slider("Prompt Guidance (CFG)", 1.0, 20.0, 8.0)
    seed = st.number_input("Seed Control", value=222, step=1)

    st.divider()
    st.subheader("Generation")
    scheduler_name = st.selectbox("Scheduler", ["Euler A", "DPM++", "DDIM"])
    num_images = st.slider("Batch Size", 1, 4, 1)

    st.divider()
    if st.button("Clear Memory"):
        logic.flush_memory()
        st.toast("Memory cleared")

tab_gen, tab_edit = st.tabs(["Generate", "Edit"])

with tab_gen:
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.subheader("Input")
        with st.form(key="gen_form"):
            prompt = st.text_area(
                "Prompt",
                "an astronaut standing on moon surface, earth visible in background, cartoon style, flat 2D illustration, clear contours",
                height=150,
            )
            neg_prompt = st.text_input(
                "Negative Prompt",
                logic.DEFAULT_NEGATIVE_PROMPT,
            )
            submit_gen = st.form_submit_button("Generate", type="primary")

        if submit_gen:
            with st.spinner("Generating image"):
                logic.flush_memory()
                generated_list = logic.generate_image(
                    pipe_txt2img,
                    prompt,
                    neg_prompt,
                    seed,
                    steps,
                    cfg,
                    num_images,
                    scheduler_name,
                )
                st.session_state["generated_images"] = generated_list
                if generated_list:
                    st.session_state["current_image"] = generated_list[0]
            st.rerun()

    with c2:
        st.subheader("Output")
        if "generated_images" in st.session_state:
            imgs = st.session_state["generated_images"]
            if len(imgs) > 1:
                cols = st.columns(2)
                for idx, img in enumerate(imgs):
                    with cols[idx % 2]:
                        st.image(img, caption=f"Image {idx + 1}", use_container_width=True)
                        if st.button(f"Select Image {idx + 1}", key=f"sel_{idx}"):
                            st.session_state["current_image"] = img
                            st.toast(f"Image {idx + 1} selected")
            elif len(imgs) == 1:
                st.image(imgs[0], caption="Result", use_container_width=True)
        else:
            st.info("Enter a prompt and press Generate.")

with tab_edit:
    if "current_image" not in st.session_state:
        st.info("Generate an image first.")
    else:
        source_img = st.session_state["current_image"]
        if isinstance(source_img, list):
            source_img = source_img[0]
            st.session_state["current_image"] = source_img

        width, height = source_img.size
        mode = st.radio("Mode", ["Inpainting", "Outpainting Zoom Out"], horizontal=True)
        st.divider()

        if mode == "Inpainting":
            col_tools, col_result = st.columns([1, 1], gap="large")
            if "canvas_key" not in st.session_state:
                st.session_state["canvas_key"] = 0

            dynamic_key = f"canvas_{st.session_state['canvas_key']}_{id(source_img)}"

            with col_tools:
                st.subheader("Draw Mask")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1.0)",
                    stroke_width=20,
                    stroke_color="#FFFFFF",
                    background_image=source_img,
                    update_streamlit=True,
                    height=height,
                    width=width,
                    drawing_mode="freedraw",
                    key=dynamic_key,
                )

            with col_result:
                st.subheader("Settings")
                with st.form("inpaint_input"):
                    edit_prompt = st.text_area(
                        "Edit Prompt",
                        "a large damaged broken satellite crashed on the moon surface, lunar craters, metallic debris, broken solar panels, exposed mechanical parts, clear silhouette, sharp focus, matching cartoon illustration style",
                        height=150,
                    )
                    strength = st.slider("Strength", 0.1, 1.0, 1.0)
                    inpaint_cfg = st.slider("Inpainting CFG", 1.0, 20.0, 12.0)
                    inpaint_steps = st.slider("Inpainting Steps", 20, 80, 50)
                    submit_inpaint = st.form_submit_button("Run Inpainting", type="primary")

                if submit_inpaint:
                    if not canvas_result.json_data or not canvas_result.json_data.get("objects"):
                        st.warning("Draw a mask first.")
                    else:
                        with st.spinner("Processing inpainting"):
                            logic.flush_memory()
                            # image_data includes the opaque background image, so its alpha
                            # channel cannot represent the drawn mask. Build the mask from
                            # pixels changed by the white canvas stroke instead.
                            painted = canvas_result.image_data[:, :, :3].astype(np.int16)
                            original = np.asarray(source_img.convert("RGB"), dtype=np.int16)
                            difference = np.max(np.abs(painted - original), axis=2)
                            mask_data = np.where(difference > 20, 255, 0).astype(np.uint8)
                            mask_image = Image.fromarray(mask_data.astype("uint8"), mode="L")
                            if mask_image.size != source_img.size:
                                mask_image = mask_image.resize(source_img.size, resample=Image.NEAREST)
                            mask_image = mask_image.filter(ImageFilter.MaxFilter(15))
                            result_img = logic.run_inpainting(
                                pipe_inpaint,
                                source_img,
                                mask_image,
                                edit_prompt,
                                strength=strength,
                                guidance_scale=inpaint_cfg,
                                num_inference_steps=inpaint_steps,
                                seed=9,
                            )
                            st.session_state["current_image"] = result_img
                            st.session_state["canvas_key"] += 1
                        st.rerun()

        else:
            col_original, col_action = st.columns([1, 1], gap="large")
            with col_original:
                st.subheader("Original")
                st.image(source_img, caption="Current image", use_container_width=True)
            with col_action:
                st.subheader("Expansion")
                with st.form("outpaint_input"):
                    out_prompt = st.text_input(
                        "Full Image Prompt",
                        "wide view of an astronaut standing on the moon, earth in the background, flat 2D cartoon illustration, clear contours",
                    )
                    submit_outpaint = st.form_submit_button("Zoom Out", type="primary")

                if submit_outpaint:
                    with st.spinner("Expanding canvas"):
                        logic.flush_memory()
                        canvas_ready, mask_ready = logic.prepare_outpainting(source_img)
                        final_result = logic.run_inpainting(pipe_inpaint, canvas_ready, mask_ready, out_prompt, 1.0)
                        st.session_state["current_image"] = final_result
                    st.rerun()
