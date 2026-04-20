from pathlib import Path

import torch
from lightglue import ALIKED as ALIKED_

from ..utils.base_model import BaseModel


class ALIKED(BaseModel):
    default_conf = {
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.05,
        "nms_radius": 2,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        conf.pop("name")
        weights = conf.pop("weights", None)

        def _sanitize_state_dict(state):
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            if isinstance(state, dict) and any(isinstance(k, str) for k in state.keys()):
                if any(k.startswith("net.") for k in state.keys()):
                    state = {k[len("net.") :] if k.startswith("net.") else k: v for k, v in state.items()}
            return state

        if weights:
            weights = str(weights)
            original_loader = torch.hub.load_state_dict_from_url

            if weights.startswith("http"):
                def _url_loader(_url, *args, map_location="cpu", **kwargs):
                    state = original_loader(weights, map_location=map_location)
                    return _sanitize_state_dict(state)

                torch.hub.load_state_dict_from_url = _url_loader
            else:
                weight_path = Path(weights)
                if not weight_path.is_absolute():
                    weight_path = Path(__file__).resolve().parents[2] / weight_path
                weight_path = weight_path.resolve()
                if not weight_path.exists():
                    raise FileNotFoundError(f"ALIKED weights not found at {weight_path}")

                def _local_loader(_url, *args, map_location="cpu", **kwargs):
                    state = torch.load(weight_path, map_location=map_location)
                    return _sanitize_state_dict(state)

                torch.hub.load_state_dict_from_url = _local_loader

            try:
                self.model = ALIKED_(**conf)
            finally:
                torch.hub.load_state_dict_from_url = original_loader
        else:
            self.model = ALIKED_(**conf)

    def _forward(self, data):
        features = self.model(data)

        return {
            "keypoints": [f for f in features["keypoints"]],
            "keypoint_scores": [f for f in features["keypoint_scores"]],
            "descriptors": [f.t() for f in features["descriptors"]],
        }

