# HuggingFace Spaces Deployment — AnyNode OS Mesh Healer
#
# To deploy on HuggingFace Spaces:
#
# 1. Create a new Space at https://huggingface.co/new-space
#    - Owner: kuparchad (or your HF username)
#    - Space name: anynode-os
#    - SDK: Docker
#    - Hardware: CPU Basic (free)
#
# 2. Clone the Space repo:
#    git clone https://huggingface.co/spaces/kuparchad/anynode-os
#
# 3. Copy these files into the Space repo:
#    cp anynode-os/hf-space/README.md anynode-os/
#    cp anynode-os/hf-space/Dockerfile anynode-os/
#    cp anynode-os/main.py anynode-os/
#    cp anynode-os/requirements.txt anynode-os/
#
# 4. Push to HuggingFace:
#    cd anynode-os && git add . && git commit -m "Deploy AnyNode OS" && git push
#
# The Space will build and deploy automatically.
# URL will be: https://kuparchad-anynode-os.hf.space
