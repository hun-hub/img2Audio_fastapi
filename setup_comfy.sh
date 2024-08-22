#!/bin/bash

# ComfyUI clone
git clone https://github.com/comfyanonymous/ComfyUI.git
cp extra_model_paths.yaml ComfyUI
cd ComfyUI/custom_nodes

# Custom nodes clone
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/crystian/ComfyUI-Crystools.git
git clone https://github.com/kijai/ComfyUI-IC-Light.git
git clone https://github.com/spacepxl/ComfyUI-Image-Filters.git
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git clone https://github.com/MinusZoneAI/ComfyUI-Kolors-MZ.git
git clone https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor.git
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git
git clone https://github.com/cubiq/ComfyUI_essentials.git
git clone https://github.com/risunobushi/comfyUI_FrequencySeparation_RGB-HSV.git
git clone https://github.com/sipherxyz/comfyui-art-venture.git
git clone https://github.com/Acly/comfyui-inpaint-nodes.git
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
git clone https://github.com/storyicon/comfyui_segment_anything.git
git clone https://github.com/rgthree/rgthree-comfy.git
git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
git clone https://github.com/Layer-norm/comfyui-lama-remover.git
git clone https://github.com/jamesWalker55/comfyui-various.git
wget -P ComfyUI-LaMA-Preprocessor/annotator/lama/models/lama/lama/ https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth
