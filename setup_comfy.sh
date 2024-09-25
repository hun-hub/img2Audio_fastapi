#!/bin/bash

pip install -r requirements.txt
# ComfyUI clone
git clone https://github.com/comfyanonymous/ComfyUI.git
cp extra_model_paths.yaml ComfyUI
cd ComfyUI
pip install -r requirements.txt
cd custom_nodes
# Custom nodes clone and install requirements if exists
repositories=(
    "https://github.com/ltdrdata/ComfyUI-Manager.git"
    "https://github.com/crystian/ComfyUI-Crystools.git"
    "https://github.com/kijai/ComfyUI-IC-Light.git"
    "https://github.com/spacepxl/ComfyUI-Image-Filters.git"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"
    "https://github.com/kijai/ComfyUI-KJNodes.git"
    "https://github.com/MinusZoneAI/ComfyUI-Kolors-MZ.git"
    "https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor.git"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"
    "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git --recursive"
    "https://github.com/cubiq/ComfyUI_essentials.git"
    "https://github.com/risunobushi/comfyUI_FrequencySeparation_RGB-HSV.git"
    "https://github.com/sipherxyz/comfyui-art-venture.git"
    "https://github.com/Acly/comfyui-inpaint-nodes.git"
    "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
    "https://github.com/storyicon/comfyui_segment_anything.git"
    "https://github.com/rgthree/rgthree-comfy.git"
    "https://github.com/WASasquatch/was-node-suite-comfyui.git"
    "https://github.com/Layer-norm/comfyui-lama-remover.git"
    "https://github.com/jamesWalker55/comfyui-various.git"
    "https://github.com/XLabs-AI/x-flux-comfyui.git"
    # Pixar
    "https://github.com/BlenderNeko/ComfyUI_Noise"
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/pythongosssss/ComfyUI-WD14-Tagger"
    "https://github.com/cubiq/ComfyUI_InstantID"
    "https://github.com/Extraltodeus/ComfyUI-AutomaticCFG"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-Gemini"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM"



    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
    "https://github.com/shadowcz007/comfyui-mixlab-nodes"
    "https://github.com/kijai/ComfyUI-Marigold"
)

for repo in "${repositories[@]}"; do
    # Clone the repository
    git clone $repo
    # Extract the folder name from the repository URL
    folder_name=$(basename $repo .git)
    # Navigate into the cloned directory
    cd $folder_name
    # Check if requirements.txt exists and install if it does
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    fi
    # Run install.py if it exists
    if [ -f install.py ]; then
        python install.py
    fi
    # Go back to the custom_nodes directory
    cd ..
done

wget -P ComfyUI-LaMA-Preprocessor/annotator/lama/models/lama/lama/ https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth
