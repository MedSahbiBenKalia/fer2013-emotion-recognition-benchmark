import os
import joblib
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

IMG_SIZE = 48
MEAN = 0.5071
STD = 0.2520

EMOTION_LABELS = {
    0: "Colere",
    1: "Degout",
    2: "Peur",
    3: "Joie",
    4: "Neutre",
    5: "Tristesse",
    6: "Surprise",
}

NOISE_LOW = 50
NOISE_HIGH = 150


def _estimate_noise(img_np: np.ndarray) -> float:
    smoothed = cv2.GaussianBlur(img_np, (5, 5), 1.0)
    residual = img_np.astype(np.float32) - smoothed.astype(np.float32)
    return float(np.std(residual))


class AdaptiveDenoiser(object):
    def __init__(self, noise_low: float = NOISE_LOW, noise_high: float = NOISE_HIGH):
        self.noise_low = noise_low
        self.noise_high = noise_high

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.uint8)
        noise = _estimate_noise(img_np)

        if noise < self.noise_low:
            denoised = img_np
        elif noise < self.noise_high:
            denoised = cv2.bilateralFilter(img_np, d=5, sigmaColor=50, sigmaSpace=50)
        else:
            denoised = cv2.fastNlMeansDenoising(
                img_np, h=10, templateWindowSize=7, searchWindowSize=21
            )

        return Image.fromarray(denoised)


class CLAHEEnhancer(object):
    def __init__(self, clip_limit: float = 1.5, grid_size: tuple = (4, 4)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        img_final = clahe.apply(img_np)
        return Image.fromarray(img_final)


class _ForceNLMDenoiser(object):
    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.uint8)
        denoised = cv2.fastNlMeansDenoising(
            img_np, h=15, templateWindowSize=7, searchWindowSize=21
        )
        return Image.fromarray(denoised)


def _build_preprocessing_steps(pipeline: str):
    if pipeline == "minimal":
        return [CLAHEEnhancer()]
    if pipeline == "light":
        return [AdaptiveDenoiser(), CLAHEEnhancer()]
    if pipeline == "aggressive":
        return [_ForceNLMDenoiser(), CLAHEEnhancer()]
    raise ValueError(
        "Unknown pipeline: '{0}'. Use minimal, light, or aggressive.".format(pipeline)
    )


def preprocess_image_for_ml(img: Image.Image, pipeline: str) -> np.ndarray:
    img_pil = img.convert("L").resize((IMG_SIZE, IMG_SIZE))
    for step in _build_preprocessing_steps(pipeline):
        img_pil = step(img_pil)

    img_array = np.array(img_pil, dtype=np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    return img_array.reshape(1, -1)


def _preprocess_for_cnn(img: Image.Image, pipeline: str):
    img_pil = img.convert("L").resize((IMG_SIZE, IMG_SIZE))
    for step in _build_preprocessing_steps(pipeline):
        img_pil = step(img_pil)
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor


def _preprocess_for_tl(img: Image.Image):
    tl_transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tl_transform(img).unsqueeze(0)


def _models_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "saved_models")


def _list_model_files(models_dir: str):
    if not os.path.isdir(models_dir):
        return []
    return sorted([f for f in os.listdir(models_dir) if f.endswith(".pkl")])


def _list_pth_files(script_dir: str):
    return sorted([f for f in os.listdir(script_dir) if f.endswith(".pth") or f.endswith(".pt")])


def _parse_model_info(filename: str):
    stem = filename[:-4]
    parts = stem.split("_")
    if len(parts) >= 2:
        pipeline = parts[-1]
        model_name = " ".join(p.title() for p in parts[:-1])
    else:
        pipeline = "light"
        model_name = stem.replace("_", " ").title()
    return model_name, pipeline


# ── CNN architecture (defined at module level so load_checkpoint can use it) ──

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 7, dropout_fc: float = 0.5):
        super().__init__()
        self.block1 = ConvBlock(in_channels=1, out_channels=64)
        self.block2 = ConvBlock(in_channels=64, out_channels=128)
        self.block3 = ConvBlock(in_channels=128, out_channels=256)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ── Robust checkpoint loader ──────────────────────────────────────────────────

def load_checkpoint(path, model_type="auto"):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        st.error(f"Failed to load checkpoint file: {e}")
        return None, None

    def _extract_state_dict(obj):
        """Pull out the model weights dict from various checkpoint formats."""
        if not isinstance(obj, dict):
            # Saved as a bare state dict / OrderedDict
            return obj
        # Common wrapper keys used by PyTorch training loops
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # If the dict itself contains tensors it is already a state dict
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        return None

    state = _extract_state_dict(ckpt)
    if state is None:
        top_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)
        st.error(
            f"Could not find a state dict inside the checkpoint.\n"
            f"Top-level keys found: {top_keys}"
        )
        return None, None

    # Detect architecture from state dict key prefixes — reliable regardless
    # of how the checkpoint was wrapped.
    keys = list(state.keys())
    is_vgg    = any(k.startswith("features.") for k in keys)
    is_resnet = any(k.startswith("layer1.")   for k in keys)
    is_cnn    = any(k.startswith("block1.")   for k in keys)

    # ── VGG-16 ───────────────────────────────────────────────────────────────
    if is_vgg or model_type == "vgg":
        vgg = models.vgg16(weights=None)
        # From notebook: avgpool replaced with AdaptiveAvgPool2d((1,1))
        # so the classifier input is 512 (not 25088).
        # Classifier: Flatten -> Linear(512,256) -> BN(256) -> ReLU -> Dropout -> Linear(256,7)
        vgg.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        vgg.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 7),
        )
        try:
            vgg.load_state_dict(state, strict=True)
            return vgg, "vgg"
        except Exception as e:
            st.warning(f"VGG strict load failed ({e}), retrying with strict=False …")
            try:
                vgg.load_state_dict(state, strict=False)
                return vgg, "vgg"
            except Exception as e2:
                st.error(f"VGG non-strict load also failed: {e2}")
                return None, None

    # ── ResNet-50 ─────────────────────────────────────────────────────────────
    elif is_resnet or model_type == "resnet":
        resnet = models.resnet50(weights=None)
        num_ftrs = resnet.fc.in_features
        # From notebook: fc.0=Linear(2048,512), fc.1=Dropout(0.5), fc.2=Linear(512,7)
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 7),
        )
        try:
            resnet.load_state_dict(state, strict=True)
            return resnet, "resnet"
        except Exception as e:
            st.warning(f"ResNet strict load failed ({e}), retrying with strict=False …")
            try:
                resnet.load_state_dict(state, strict=False)
                return resnet, "resnet"
            except Exception as e2:
                st.error(f"ResNet non-strict load also failed: {e2}")
                return None, None

    # ── Custom CNN ────────────────────────────────────────────────────────────
    elif is_cnn or model_type == "cnn":
        model = CustomCNN(num_classes=7)
        try:
            model.load_state_dict(state, strict=True)
            return model, "cnn"
        except Exception as e:
            st.warning(f"CNN strict load failed ({e}), retrying with strict=False …")
            try:
                model.load_state_dict(state, strict=False)
                return model, "cnn"
            except Exception as e2:
                st.error(f"CNN non-strict load also failed: {e2}")
                return None, None

    # ── Unknown ───────────────────────────────────────────────────────────────
    else:
        st.error(
            "Could not detect architecture from checkpoint key names.\n"
            f"First 10 keys: {keys[:10]}\n\n"
            "Expected prefixes:  'features.*' → VGG  |  'layer1.*' → ResNet  |  'block1.*' → CustomCNN"
        )
        return None, None


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="FER2013 Model Tester", layout="centered")

st.title("FER2013 Model Tester")

models_dir = _models_dir()
model_files = _list_model_files(models_dir)

st.sidebar.header("Models")
if model_files:
    model_choice = st.sidebar.selectbox("ML model", model_files)
    model_name, pipeline = _parse_model_info(model_choice)
    st.sidebar.caption("Pipeline: {0}".format(pipeline))
else:
    model_choice = None
    st.sidebar.warning("No saved ML models found in saved_models/.")

st.sidebar.header("Deep Learning")
pth_files = _list_pth_files(os.path.dirname(os.path.abspath(__file__)))

dl_choice = None
if pth_files:
    st.sidebar.write("Available .pth files in project:")
    dl_choice = st.sidebar.selectbox("DL checkpoint", ["-- none --"] + pth_files)
else:
    st.sidebar.info("No .pth files found in project root.")

uploaded = st.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])

# ── Classic ML inference ──────────────────────────────────────────────────────
if uploaded and model_choice:
    image = Image.open(uploaded)
    st.image(image, caption="Input", use_container_width=True)

    model_path = os.path.join(models_dir, model_choice)
    model = joblib.load(model_path)

    _, pipeline = _parse_model_info(model_choice)
    features = preprocess_image_for_ml(image, pipeline=pipeline)

    pred = model.predict(features)
    pred_label = EMOTION_LABELS.get(int(pred[0]), str(pred[0]))

    st.subheader("Prediction")
    st.write("Model: {0}".format(model_choice))
    st.write("Label: {0}".format(pred_label))

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        proba_dict = {
            EMOTION_LABELS[i]: float(proba[i]) for i in range(len(proba))
        }
        st.subheader("Probabilities")
        st.json(proba_dict)
else:
    if not uploaded or not model_choice:
        st.info("Upload an image and select a saved ML model to run inference.")

# ── Deep Learning inference ───────────────────────────────────────────────────
if uploaded and dl_choice and dl_choice != "-- none --":
    image = Image.open(uploaded).convert("RGB")
    st.header("Deep Learning Inference")
    st.image(image, caption="Input (DL)", use_container_width=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pth_path = os.path.join(script_dir, dl_choice)

    # Show classifier/fc key shapes so architecture mismatches are obvious
    try:
        _ckpt_raw = torch.load(pth_path, map_location="cpu", weights_only=False)
        _sd = _ckpt_raw
        for _k in ("model_state_dict", "state_dict", "model_state", "model"):
            if isinstance(_ckpt_raw, dict) and _k in _ckpt_raw:
                _sd = _ckpt_raw[_k]
                break
        if isinstance(_sd, dict):
            with st.expander("🔍 Checkpoint key preview (for debugging)"):
                _head_keys = [k for k in _sd.keys() if any(x in k for x in ("classifier", "fc.", "block"))][:20]
                st.code("\n".join(f"{k}: {tuple(_sd[k].shape)}" for k in _head_keys))
    except Exception:
        pass

    model_obj, model_kind = load_checkpoint(pth_path)

    if model_obj is None:
        st.error("Failed to load model from checkpoint.")
    else:
        model_obj.eval()
        with torch.no_grad():
            if model_kind == "cnn":
                tensor = _preprocess_for_cnn(image, pipeline="light")
                out = model_obj(tensor)
            elif model_kind in ("vgg", "resnet"):
                tensor = _preprocess_for_tl(image)
                out = model_obj(tensor)
            else:
                st.error("Unknown model kind.")
                out = None

            if out is not None:
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())
                pred_label = EMOTION_LABELS.get(pred_idx, str(pred_idx))

                st.subheader("DL Prediction")
                st.write(f"Model file: {dl_choice}")
                st.write(f"Architecture detected: {model_kind.upper()}")
                st.write(f"Predicted: **{pred_label}** (class {pred_idx})")

                st.subheader("Probabilities")
                st.json({EMOTION_LABELS[i]: float(probs[i]) for i in range(len(probs))})