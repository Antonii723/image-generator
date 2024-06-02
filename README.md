# Image Generator Flask App

# Description 

This system should let users pick one or more models, enter text prompts, and get images from the selected models. This project will show how different models can work together in one system.

# Objectives

**1. Create an easy-to-use interface where users can select and run multiple image generation models.** 
**2. Integrate at least five different open-source image generation models:** 

* Stable Diffusion (SD)
* DeepFloyd IF
* DreamShaper
* OpenJourney
* PixArt-Alpha

**3. Make sure the system can handle multiple prompts and generate outputs efficiently.** 
**4. Collect performance metrics for each model, such as the time taken to generate an image and other relevant statistics.** 

# Deliverables

**1. User Interface (UI)** 

* A web-based interface that allows users to:
* Select one or more image generation models.
* Enter text prompts.
* View the generated images from each selected model.
* Focus on basic design and functionality.

**2. Model Integration** 

* Add the following image generation models:
* Stable Diffusion (SD)
* DeepFloyd IF
* DreamShaper
* OpenJourney
* PixArt-Alpha
* Ensure users can select and run each model through the interface.

**3. Backend System** 

* Develop a backend to handle requests from the UI and interact with the models.
* Efficiently manage input prompts and model outputs.
* Log and handle errors.
* Collect performance data such as:

    - Time taken to generate each image.
    - System resource usage (CPU, GPU, memory).
    - Success rate of image generation.
    - Number of requests handled per model.
    - Average time per request.
    - Frequency of model-specific errors.
    - Peak memory usage during image generation.
    - User satisfaction rating (optional: collected through UI feedback).

# Installation 

**1. Download repository**
```
git clone https://github.com/Antonii723/image-generator.git
```
**2. Install the Python v3.11.2**

**3. Install requirements** 
```
pip v24.0
python.exe -m pip install --upgrade pip

cd image-generation
pip install -r requirement.txt
```

**4. Run WebApp** 
```
python app.py

(Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe)

Please run this app on your web browser.
http://localhost:5000/
```

