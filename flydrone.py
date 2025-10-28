import airsim
import time
import os
import io
import pickle
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# --- 1. Configuration ---

MODEL_BASE_DIR = r"C:\Dev\blip_model\firstapp"

IMAGE_SAVE_PATH = "captured_scene.jpg"

BLIP_MODEL_DIR = os.path.join(MODEL_BASE_DIR, "models", "blip_model", "blip_scene_finetuned_2")
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_ADAPTER_DIR = os.path.join(MODEL_BASE_DIR, "models", "llm_model_folder", "tinyllama_adapter")
VECTORIZER_PATH = os.path.join(MODEL_BASE_DIR, "models", "tfidf_vectorizer.pkl")
ML_MODEL_PATH = os.path.join(MODEL_BASE_DIR, "models", "risk_model.pkl")

# --- DRONE SETTINGS ---
TAKEOFF_ALT = -1.0
VELOCITY = 1.0
RISK_THRESHOLD = 2.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Load Models ---

def load_blip_model():
    print("Loading BLIP model...")
    try:
        processor = BlipProcessor.from_pretrained(BLIP_MODEL_DIR)
        model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_DIR,
            load_in_8bit=True,
            device_map="auto"
        )
        model.eval()
        print("BLIP model loaded successfully.")
        return processor, model
    except Exception as e:
        print(f"Error loading BLIP model: {e}")
        return None, None


def load_llm_model():
    print("Loading LLM model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            load_in_8bit=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, LLM_ADAPTER_DIR)
        model.eval()
        print("LLM model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        return None, None


def load_ml_models():
    print("Loading ML models...")
    try:
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(ML_MODEL_PATH, 'rb') as f:
            ml_model = pickle.load(f)
        print("Risk prediction ML models loaded successfully.")
        return vectorizer, ml_model
    except Exception as e:
        print(f"Error loading ML models: {e}")
        return None, None


blip_processor, blip_model = load_blip_model()
llm_tokenizer, llm_model = load_llm_model()
vectorizer, ml_risk_model = load_ml_models()

# --- 3. Pipeline ---

def generate_caption(image, processor, model):
    """BLIP: Generate caption from image"""
    if processor is None or model is None:
        return None
    try:
        print("Sending image to BLIP for caption generation...")
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=inputs.pixel_values, max_length=64, num_beams=4, do_sample=False
            )
        caption = processor.decode(generated_ids.squeeze(0), skip_special_tokens=True)
        print("Received caption from BLIP.")
        return caption
    except Exception as e:
        print(f"Caption generation failed: {e}")
        return None


def predict_next_scene(model, tokenizer, description, max_tokens=120):
    """TinyLlama: Predict next probable scene"""
    if tokenizer is None or model is None:
        return None
    prompt = f"Instruction: Predict the next probable scene description given the current scene.\nInput: {description}\nOutput:"
    try:
        print("Sending description to LLM for next scene prediction...")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print("Received next scene prediction from LLM.")
        return response.strip()
    except Exception as e:
        print(f"LLM prediction failed: {e}")
        return None


def predict_risk(description, next_scene, vectorizer, ml_model):
    """Predict risk level using ML model"""
    if vectorizer is None or ml_model is None:
        return None, None
    combined_text = (description or "") + " " + (next_scene or "")
    try:
        print("Sending text to ML model for risk prediction...")
        vec = vectorizer.transform([combined_text])
        risk = ml_model.predict(vec)[0]
        risk_percent = np.clip(risk / 10 * 100, 0, 100)
        print("Received risk prediction from ML model.")
        return risk, risk_percent
    except Exception as e:
        print(f"Risk prediction failed: {e}")
        return None, None

# --- 4. Drone Operations ---

def capture_and_process_image(client):
    print("DRONE1: Capturing image...")
    response = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
    ], vehicle_name="Drone1")

    if not response or not response[0].image_data_uint8:
        print("ERROR: Failed to get valid image data.")
        return None

    image = Image.open(io.BytesIO(response[0].image_data_uint8))
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(IMAGE_SAVE_PATH, format="JPEG")
    print(f"Image saved to {IMAGE_SAVE_PATH}")

    desc = generate_caption(image, blip_processor, blip_model)
    if not desc:
        return None
    print(f"Current Scene Description: {desc}")

    next_scene = predict_next_scene(llm_model, llm_tokenizer, desc)
    if not next_scene:
        return None
    print(f"Next Scene Prediction: {next_scene}")

    risk_score, risk_percent = predict_risk(desc, next_scene, vectorizer, ml_risk_model)
    if risk_score is None:
        return None

    print("\n--- AI Pipeline Results ---")
    print(f"Predicted Risk Score: {risk_score:.2f}/10 ({risk_percent:.1f}%)")
    print("---------------------------\n")
    return risk_score


def main_airsim_ai():
    if not all([blip_model, llm_model, vectorizer, ml_risk_model]):
        print("Model loading failed. Exiting.")
        return

    client1 = airsim.MultirotorClient()
    client2 = airsim.MultirotorClient()
    drone2_active = False

    try:
        print("Connecting to AirSim...")
        client1.confirmConnection()
        client2.confirmConnection()

        client1.enableApiControl(True, "Drone1")
        client1.armDisarm(True, "Drone1")
        print("Drone1 connected and armed.")

        print("Drone1 taking off...")
        client1.takeoffAsync(vehicle_name="Drone1").join()
        client1.moveToZAsync(TAKEOFF_ALT, VELOCITY, vehicle_name="Drone1").join()
        print(f"Drone1 positioned at Z = {TAKEOFF_ALT}")

        risk_score = capture_and_process_image(client1)
        if risk_score is None:
            print("Risk analysis failed.")
            return

        if risk_score > RISK_THRESHOLD:
            drone2_active = True
            print(f"Risk ({risk_score:.2f}) detected. Launching Drone2...")

            client2.enableApiControl(True, "Drone2")
            client2.armDisarm(True, "Drone2")

            d1_state = client1.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position
            d1_x, d1_y, d1_z = d1_state.x_val, d1_state.y_val, d1_state.z_val

            target_x = d1_x - 5.0
            target_y = d1_y - 5.0
            target_z = d1_z - 1.0  # 1 meter above Drone1

            print(f"Drone1 position: X:{d1_x:.2f}, Y:{d1_y:.2f}, Z:{d1_z:.2f}")
            print(f"Drone2 target position: X:{target_x:.2f}, Y:{target_y:.2f}, Z:{target_z:.2f}")

            print("Drone2 taking off to target altitude...")
            client2.takeoffAsync(vehicle_name="Drone2").join()
            client2.moveToZAsync(target_z, 1.0, vehicle_name="Drone2").join()

            print("Drone2 flying to target position...")
            client2.moveToPositionAsync(target_x, target_y, target_z, 2.0, vehicle_name="Drone2").join()
            print("Drone2 reached target position.")

            time.sleep(3)

            response = client2.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
            ], vehicle_name="Drone2")
            if response and response[0].image_data_uint8:
                img = Image.open(io.BytesIO(response[0].image_data_uint8))
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save("captured_scene_drone2_verification.jpg", format="JPEG")
                print("Drone2 verification image saved.")

            print("Drone2 returning to launch point...")
            client2.moveToPositionAsync(0, 0, -1, 2.0, vehicle_name="Drone2").join()
            time.sleep(1)
            client2.landAsync(vehicle_name="Drone2").join()
            print("Drone2 landed successfully.")

        else:
            print(f"Risk below threshold ({risk_score:.2f}). Drone2 on standby.")

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        print("Landing and cleanup...")
        try:
            client1.landAsync(vehicle_name="Drone1").join()
            print("Drone1 landed successfully.")
        except:
            pass
        client1.armDisarm(False, "Drone1")
        client1.enableApiControl(False, "Drone1")

        if drone2_active:
            try:
                client2.armDisarm(False, "Drone2")
                client2.enableApiControl(False, "Drone2")
            except:
                pass
        print("Mission complete. Drones disarmed and control released.")


if __name__ == "__main__":
    main_airsim_ai()
