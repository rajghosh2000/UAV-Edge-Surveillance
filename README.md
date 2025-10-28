# UAV Edge Surveillance

## Overview
This project demonstrates an **AI-powered UAV (Drone) Surveillance System** that integrates computer vision, large language models (LLMs), and risk prediction using machine learning. The system operates inside a simulated drone environment built using **Microsoft AirSim** and **Unreal Engine 4.27.2**, where multiple drones perform coordinated tasks based on AI predictions.

---

## System Architecture

### Components
1. **BLIP Model (from Salesforce)**  
   - Fine-tuned to understand drone-captured images and generate **scene captions**.

2. **LLM (Large Language Model)**  
   - Takes the scene caption as input and predicts the **next possible scene**.

3. **ML Risk Prediction Model**  
   - Trained to predict the **risk score** from the LLM-generated description.

---

## Workflow

1. **Drone 1** takes off and captures an image from the simulation, and this image is sent to the blip model.  
2. The **BLIP model** generates a **scene description** from the image.  
3. This description is sent to the **LLM**, which predicts the **next possible scene**.  
4. The predicted scene is analyzed by the **ML model**, which outputs a **risk score (0–10)**.  
5. If the **risk exceeds the threshold**, **Drone 2** is deployed to verify the area.  
6. Both drones return to their launch positions after the mission.

---

## Setup Instructions

### 1. Install Unreal Engine and AirSim
- Install **Visual Studio 2022**.  Make sure to select Desktop Development with C++ and Windows 10 SDK 10.0.19041 and select the latest .NET Framework SDK under the 'Individual Components' tab while installing VS 2022.
- Download **Unreal Engine 4.27.2** from Epic Games Launcher.
- Clone **Microsoft AirSim** from GitHub and build it.

### 2. Create Unreal Project
- Create a new Unreal project (C++ template).  
- Add **AirSim Plugin** to your project.  
- Open the `.sln` file in **Visual Studio**.  
- Select `DebugGame Editor` → click **Local Windows Debugger** to open Unreal.  
- In Unreal → **World Settings → Game Mode** → set to `AirSimGameMode`.

### 3. Connect AirSim with Python
- Update your **settings.json** (in `Documents/AirSim/`) to configure drone vehicles.
- Launch the Unreal project and click **Play**.

### 4. Run the Drone Script
Run the following inside your virtual environment:

```bash
python flydrone.py
```

This script:
- Connects to AirSim
- Flies **Drone1**, captures images
- Sends them to models for scene and risk analysis
- Launches **Drone2** if high risk is detected

---

## Project Structure

```
UAV-Edge-Surveillance/
│
├── flydrone.py               # Main drone control and AI integration script
├── models/
│   ├── blip_model/
|       ├── blip_scene_finetuned_2/  # Fine-tuned BLIP model
│   ├── llm_model_folder/            # Fine-tuned LLM
│   ├── risk_model.pkl        # Trained ML risk prediction model
│   └── tfidf_vectorizer.pkl  # TF-IDF vectorizer for text features
├── settings.json             # AirSim configuration
└── README.md                 # Project documentation
```

---

## Dependencies

- Python 3.10+  
- PyTorch  
- Transformers  
- scikit-learn  
- AirSim Python API  
- Pillow, NumPy, torch, peft

Install using:
```bash
pip install -r requirements.txt
```

---

## Usage Example

```bash
(venv) C:\Dev\blip_model\firstapp> python flydrone.py
```

Output Example:
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
- Ensure Unreal Engine and AirSim are **running before executing** the script.
- If risk model or BLIP model paths differ, update them in `flydrone.py`.
- To suppress warnings, run Python with:
  ```bash
  python -W ignore flydrone.py
  ```

---

## Credits

---

## License
