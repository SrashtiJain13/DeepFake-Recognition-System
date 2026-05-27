from pathlib import Path
import sys

import torch
import timm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import preprocess_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = PROJECT_ROOT / "image_model_effnet.pth"
_model = None


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2,
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_model():
    global _model

    if _model is None:
        _model = load_model()

    return _model


def predict_image(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_tensor = preprocess_image(str(image_path))
    if image_tensor is None:
        return {
            "label": "NO FACE DETECTED",
            "real_prob": 0.0,
            "fake_prob": 0.0,
        }

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    model = get_model()

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0].detach().cpu().tolist()

    real_prob = float(probs[0]) * 100
    fake_prob = float(probs[1]) * 100
    label = "FAKE" if fake_prob > real_prob else "REAL"

    return {
        "label": label,
        "real_prob": round(real_prob, 2),
        "fake_prob": round(fake_prob, 2),
    }


if __name__ == "__main__":
    user_image_path = input("Enter image path: ").strip()

    try:
        result = predict_image(user_image_path)
        print("\nPrediction Result")
        print("-----------------")
        print(f"Prediction: {result['label']}")
        print(f"Real: {result['real_prob']:.2f}%")
        print(f"Fake: {result['fake_prob']:.2f}%")
    except Exception as exc:
        print(f"Error: {exc}")
