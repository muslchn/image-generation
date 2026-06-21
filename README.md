# Image Generation Submission - BFGAI

This repository contains a Dicoding **Belajar Fundamental Generative AI** submission for the Image Generation project. It demonstrates a Stable Diffusion workflow for text-to-image generation, image manipulation, and an interactive Streamlit application.

## Overview

The project is split into two notebooks:

- `Pipeline_submission_BFGAI_Muslichin.ipynb`  
  Contains the image generation experiments, parameter comparisons, scheduler comparisons, inpainting, and outpainting workflow.
- `Streamlit_submission_BFGAI_Muslichin.ipynb`  
  Creates and runs the Streamlit application used for interactive image generation and editing.

The final submission package also requires a real demo video named `video_demo_aplikasi_BFGAI.mp4`. This video must be recorded after the Streamlit application runs successfully.

## Main Capabilities

The implementation covers the required image generation workflow:

- Text-to-image generation with `runwayml/stable-diffusion-v1-5`.
- Reproducible generation using prompt, negative prompt, and seed.
- Advanced generation controls using guidance scale and inference steps.
- Visual comparison for guidance scale values.
- Visual comparison for low and high inference step values.
- Batch inference with four generated images displayed in a 2x2 grid.
- Scheduler switching with Euler A, DPM++, and DDIM.
- Inpainting with `runwayml/stable-diffusion-inpainting`.
- Manual mask creation for inpainting.
- Segmentation-based automatic masking.
- Outpainting by expanding the canvas.
- Zoom-out style outpainting.
- Streamlit interface for generation, batch output, scheduler selection, memory cleanup, inpainting, and outpainting.

## Required Files

The root submission folder should contain these files before creating the final ZIP:

```text
Pipeline_submission_BFGAI_Muslichin.ipynb
Streamlit_submission_BFGAI_Muslichin.ipynb
video_demo_aplikasi_BFGAI.mp4
requirements.txt
```

`video_demo_aplikasi_BFGAI.mp4` is not a placeholder file. It must be an actual screen recording of the Streamlit application running.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Recommended runtime:

- Google Colab or Kaggle Notebook
- GPU enabled
- Python 3.10 or newer

Stable Diffusion is resource intensive. A GPU runtime is strongly recommended for both notebooks.

## Running the Pipeline Notebook

Open:

```text
Pipeline_submission_BFGAI_Muslichin.ipynb
```

Recommended workflow:

1. Open the notebook in Google Colab or Kaggle.
2. Enable GPU runtime.
3. Run the dependency installation cell.
4. Run the Stable Diffusion pipeline loading cell.
5. Run the text-to-image generation cells.
6. Run the guidance scale and inference step comparison cells.
7. Run the batch inference and scheduler comparison cells.
8. Run the inpainting, automatic masking, outpainting, and zoom-out cells.
9. Save the executed notebook so all output images are visible to the reviewer.

## Running the Streamlit Application

Open:

```text
Streamlit_submission_BFGAI_Muslichin.ipynb
```

Recommended workflow:

1. Open the notebook in Google Colab or Kaggle.
2. Enable GPU runtime.
3. Run the dependency installation cell.
4. Run the cell that writes `logic.py`.
5. Run the cell that writes `app.py`.
6. Enter your private Ngrok auth token only when the hidden runtime prompt appears.
7. Run the Streamlit launch cell.
8. Run the Ngrok public URL cell.
9. Open the generated public URL.
10. Test the image generation flow from the Streamlit interface.

Do not commit, record, or share your Ngrok auth token. If a token was ever stored in a notebook or exposed elsewhere, revoke it and generate a replacement before continuing.

## Streamlit Demo Checklist

When testing the app, verify these interactions:

- The app opens successfully in the browser.
- Prompt and negative prompt inputs are editable.
- Quality steps and CFG sliders work.
- Scheduler selection works for Euler A, DPM++, and DDIM.
- Batch size can be set from 1 to 4.
- The Generate button produces and displays image output.
- The Clear Memory button can be clicked without crashing the app.
- Inpainting accepts a drawn mask and produces an edited image.
- Outpainting expands the image canvas and produces a zoomed-out result.

## Creating the Demo Video

Record a 1-5 minute screen recording of the running Streamlit app. The video should show the application interface, not only the notebook.

Suggested recording sequence:

1. Open the Streamlit app through the Ngrok URL.
2. Enter a prompt and negative prompt.
3. Adjust `Quality Steps`, `Creativity (CFG)`, seed, scheduler, and batch size.
4. Click Generate.
5. Wait until generated images appear.
6. Show the 2x2 grid if using batch generation.
7. Click Clear Memory.
8. Open the Edit tab.
9. Draw a mask and run inpainting.
10. Run outpainting or zoom-out.

Save the video with this exact filename:

```text
video_demo_aplikasi_BFGAI.mp4
```

Before packaging, confirm that the video:

- Uses `.mp4` format.
- Has a non-zero file size.
- Can be played normally.
- Shows generated output.
- Does not reveal Ngrok tokens, passwords, API keys, or other private credentials.

## Final ZIP Submission

Create the final ZIP from the repository root:

```bash
zip -r BFGAI_Muslichin.zip \
  Pipeline_submission_BFGAI_Muslichin.ipynb \
  Streamlit_submission_BFGAI_Muslichin.ipynb \
  video_demo_aplikasi_BFGAI.mp4 \
  requirements.txt
```

The ZIP should contain only the required files at the top level:

```text
BFGAI_Muslichin.zip
├── Pipeline_submission_BFGAI_Muslichin.ipynb
├── Streamlit_submission_BFGAI_Muslichin.ipynb
├── video_demo_aplikasi_BFGAI.mp4
├── requirements.txt
```

## Pre-Submission Checklist

- [ ] `Pipeline_submission_BFGAI_Muslichin.ipynb` has been executed.
- [ ] Pipeline notebook outputs are visible and saved.
- [ ] `Streamlit_submission_BFGAI_Muslichin.ipynb` has been executed.
- [ ] Streamlit app opens through Ngrok.
- [ ] Text-to-image generation works from the app.
- [ ] Batch generation and scheduler selection work.
- [ ] Inpainting and outpainting features have been tested.
- [ ] Demo video is named `video_demo_aplikasi_BFGAI.mp4`.
- [ ] Demo video shows the app running and generating images.
- [ ] `requirements.txt` is included in the ZIP.
- [ ] No credentials are included in the notebooks, video, or ZIP file.

## Notes

- If the runtime runs out of VRAM, clear memory from the app or restart the notebook runtime.
- If Ngrok returns a limit or tunnel error, close active tunnels and start a new one.
- Save notebooks after execution so the reviewer can inspect outputs without rerunning the full workflow.
