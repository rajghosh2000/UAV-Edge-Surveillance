# UAV Edge Surveillance

## Overview
This project demonstrates a **UAV (Drone) Surveillance System** that integrates computer vision, language modeling, and risk prediction using machine learning.  
The system operates inside a simulated drone environment built using **Microsoft AirSim** and **Unreal Engine 4.27.2**, where multiple drones perform coordinated tasks based on model predictions.

---

## System Architecture

### Components
1. **BLIP Model (from Salesforce)**  
   - Fine-tuned to understand drone-captured images and generate **scene captions**.

2. **TinyLlama (Language Model)**  
   - A lightweight text model that takes the scene caption as input and predicts the **next possible scene**.

3. **ML Regression Model**  
   - A machine learning regression model trained to predict the **risk score** from the generated description.

---

## Workflow

1. **Drone 1** takes off and captures an image from the simulation; the image is sent to the BLIP model.  
2. The **BLIP model** generates a **scene description** from the image.  
3. This description is passed to the **TinyLlama model**, which predicts the **next possible scene**.  
4. The **ML regression model** analyzes the text and outputs a **risk score (0–10)**.  
5. If the **risk exceeds the threshold**, **Drone 2** is deployed to verify the area.  
6. After verification, both drones **return to their launch positions**.

---

## Setup Instructions

### 1. Install Unreal Engine and AirSim
- Install **Visual Studio 2022**.  
  Select *Desktop Development with C++*, *Windows 10 SDK (10.0.19041)*, and the latest *.NET Framework SDK* during setup.
- Download **Unreal Engine 4.27.2** from Epic Games Launcher.
- Clone and build **Microsoft AirSim** from GitHub.

### 2. Create Unreal Project
- Create a new Unreal project (C++ template).  
- Add the **AirSim Plugin** to your project.  
- Open the `.sln` file in **Visual Studio**.  
- Select **DebugGame Editor** and click **Local Windows Debugger** to open Unreal.  
- In Unreal → **World Settings → Game Mode**, set it to `AirSimGameMode`.

### 3. Connect AirSim with Python
- Update `Documents/AirSim/settings.json` to define your drone vehicles.  
- Launch the Unreal project and click **Play**.

### 4. Run the Drone Script
Run the following inside your virtual environment:

```bash
python flydrone.py
```

This script:
- Connects to AirSim  
- Flies **Drone 1** and captures an image  
- Processes it through BLIP → TinyLlama → ML regression  
- Launches **Drone 2** if the risk exceeds a threshold  

---

## Project Structure

```
UAV-Edge-Surveillance/
│
├── flydrone.py               # Drone control and model integration script
├── models/
│   ├── blip_model/
│   │   └── blip_scene_finetuned_2/  # Fine-tuned BLIP model
│   ├── llm_model_folder/            # Fine-tuned TinyLlama model
│   ├── risk_model.pkl        # ML regression model for risk prediction
│   └── tfidf_vectorizer.pkl  # TF-IDF vectorizer for text processing
├── settings.json             # AirSim configuration
└── README.md                 # Documentation
```

---

## Dependencies

- Python 3.10+  
- PyTorch  
- Transformers  
- scikit-learn  
- AirSim Python API  
- Pillow, NumPy, torch, peft  

---

## Usage Example

```bash
(venv) C:\Dev\blip_model\firstapp> python flydrone.py
```

Example output:
```
Drone1 capturing image...
Scene Caption: "a man standing near a table with tools"
Next Scene: "the man accidentally drops a tool on the ground"
Predicted Risk: 4.6 / 10
Drone2 deployed for verification...
Mission complete.
```

---

## Notes
- Ensure **Unreal Engine and AirSim** are running before executing the script.  
- Update model paths in `flydrone.py` if they differ from your directory structure.  
- To suppress warnings, run:
  ```bash
  python -W ignore flydrone.py
  ```

---

## Credits

---

## License
