import argparse
import json
import h5py
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPVisionModelWithProjection


def parse_args_training(input_args=None):

    parser = argparse.ArgumentParser()

    # pretrained weights
    parser.add_argument("--sd_path", required=True, help="Path to SD-Turbo")
    parser.add_argument("--elic_path", required=True, help="Path to pretrained ELIC model")
    parser.add_argument("--codec_path", help="Path to pretrained StableCodec weights", default=None)

    # dataset
    parser.add_argument("--train_dataset", required=True, help="Path to training dataset (hdf5)")
    parser.add_argument("--test_dataset", required=True, help="Path to test dataset (Kodak)")
    parser.add_argument("--caption_json", default="data/train", help="Caption source. Can be a single json file, a directory containing dataset jsons, or a mapping json.")
    parser.add_argument("--test_caption_json", default="data/test/kodak_caption.json", help="Path to validation image-caption json file.")

    # loss function
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.1, type=float)
    parser.add_argument("--lambda_clip", default=0.1, type=float)
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)
    parser.add_argument("--lambda_rate", required=True, default=0.5, type=float)

    # model details
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=160)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--train_patch_size", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_steps", type=int, default=120000)
    parser.add_argument("--checkpointing_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_val", default=True)
    parser.add_argument("--save_num", type=int, default=10, help="Number of visual samples to save")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class H5CaptionDataset(Dataset):
    def __init__(self, path, caption_json, transform=None):
        self.file_path = path
        self.caption_json = caption_json
        self.transform = transform
        self.dataset = h5py.File(self.file_path, 'r')
        self.caption_root = Path(caption_json)
        self.caption_mapping = self._load_caption_mapping()
        self.caption_cache = {}
        self.samples = []

        with h5py.File(self.file_path, 'r') as file:
            for key in file.keys():
                entry = file[key]
                if not isinstance(entry, h5py.Group) or "image" not in entry:
                    raise ValueError(
                        "Caption training expects the HDF5 layout written by build_h5_caption.py. "
                        "Please rebuild the training dataset with image metadata."
                    )

                image = entry["image"]
                if image.shape[0] >= 512 and image.shape[1] >= 512:
                    image_name = self._decode_string(entry.attrs.get("image_name", key))
                    source_dir = self._decode_string(entry.attrs.get("source_dir", ""))
                    self.samples.append({
                        "key": key,
                        "image_name": image_name,
                        "source_dir": source_dir,
                    })

    def _load_caption_mapping(self):
        if not self.caption_root.exists():
            raise FileNotFoundError(f"Caption source not found: {self.caption_root}")

        if self.caption_root.is_dir():
            return None

        with self.caption_root.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError("Caption source json must be a dictionary.")

        if len(payload) > 0 and all(self._is_caption_payload(value) for value in payload.values()):
            return {"__single__": self.caption_root}

        return payload

    def _decode_string(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return value.decode("utf-8")
        return str(value)

    def _is_caption_payload(self, payload):
        if isinstance(payload, str):
            return True
        if isinstance(payload, list):
            return all(isinstance(item, str) for item in payload)
        if isinstance(payload, dict):
            return any(isinstance(payload.get(field), str) for field in ("caption", "text", "prompt"))
        return False

    def _normalize_dataset_name(self, source_dir):
        source_path = Path(source_dir)
        parts = [part.lower() for part in source_path.parts]

        if "flickr2k" in parts:
            return "flickr2k"
        if "div2k" in parts:
            return "div2k"
        if "clic2020train" in parts:
            return "clic2020train"
        if "clic2020val" in parts:
            return "clic2020val"
        if "clic2021test" in parts:
            return "clic2021test"
        if "lsdir10k" in parts or "lsdir" in parts:
            return "lsdir10k"

        return source_path.name.lower()

    def _resolve_caption_file(self, source_dir):
        dataset_name = self._normalize_dataset_name(source_dir)

        if self.caption_mapping is not None:
            if "__single__" in self.caption_mapping:
                return self.caption_mapping["__single__"]

            for candidate in (dataset_name, Path(source_dir).name, str(source_dir)):
                if candidate in self.caption_mapping:
                    caption_file = Path(self.caption_mapping[candidate])
                    if not caption_file.is_absolute():
                        caption_file = self.caption_root.parent / caption_file
                    return caption_file

            raise KeyError(f"No caption json mapping found for source_dir={source_dir}")

        candidate_files = [
            self.caption_root / f"{dataset_name}.json",
            self.caption_root / f"{Path(source_dir).name}.json",
        ]
        for candidate in candidate_files:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find caption json for source_dir={source_dir}. "
            f"Tried: {', '.join(str(path) for path in candidate_files)}"
        )

    def _load_captions_for_source(self, source_dir):
        source_key = str(source_dir)
        if source_key in self.caption_cache:
            return self.caption_cache[source_key]

        caption_file = self._resolve_caption_file(source_dir)
        with caption_file.open("r", encoding="utf-8") as f:
            captions = json.load(f)

        if not isinstance(captions, dict):
            raise ValueError(f"Caption json must be a dictionary: {caption_file}")

        self.caption_cache[source_key] = captions
        return captions

    def _extract_caption_text(self, payload):
        if isinstance(payload, str):
            return payload

        if isinstance(payload, list):
            if not payload:
                raise ValueError("Caption list must not be empty")
            first_item = payload[0]
            if not isinstance(first_item, str):
                raise TypeError("Caption list items must be strings")
            return first_item

        if isinstance(payload, dict):
            for field in ("caption", "text", "prompt"):
                value = payload.get(field)
                if isinstance(value, str) and value:
                    return value
            raise ValueError("Caption dict must contain one of: caption, text, prompt")

        raise TypeError(f"Unsupported caption payload type: {type(payload).__name__}")

    def _resolve_caption(self, image_name, source_dir):
        captions = self._load_captions_for_source(source_dir)

        if image_name in captions:
            return self._extract_caption_text(captions[image_name])

        image_stem = Path(image_name).stem
        if image_stem in captions:
            return self._extract_caption_text(captions[image_stem])

        raise KeyError(
            f"Missing caption for image. Tried keys: {image_name}, {image_stem}. "
            f"source_dir={source_dir}, caption_source={self.caption_json}"
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.dataset[sample["key"]]["image"][:]
        caption = self._resolve_caption(sample["image_name"], sample["source_dir"])
        # print(sample['image_name'])
        # print(caption)

        if self.transform:
            image = self.transform(image)
        return image, caption

    def __len__(self):
        return len(self.samples)


class CaptionImageDataset(Dataset):
    def __init__(self, image_dir, caption_json, transform=None):
        self.image_dir = Path(image_dir)
        self.caption_json = Path(caption_json)
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Validation image directory not found: {self.image_dir}")
        if not self.caption_json.exists():
            raise FileNotFoundError(f"Validation caption json not found: {self.caption_json}")

        with self.caption_json.open("r", encoding="utf-8") as f:
            self.captions = json.load(f)

        if not isinstance(self.captions, dict):
            raise ValueError(f"Validation caption json must be a dictionary: {self.caption_json}")

        self.image_paths = []
        for path in sorted(self.image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                self.image_paths.append(path)

        if len(self.image_paths) == 0:
            raise ValueError(f"No validation images found in {self.image_dir}")

    def _extract_caption_text(self, payload):
        if isinstance(payload, str):
            return payload

        if isinstance(payload, list):
            if not payload:
                raise ValueError("Caption list must not be empty")
            first_item = payload[0]
            if not isinstance(first_item, str):
                raise TypeError("Caption list items must be strings")
            return first_item

        if isinstance(payload, dict):
            for field in ("caption", "text", "prompt"):
                value = payload.get(field)
                if isinstance(value, str) and value:
                    return value
            raise ValueError("Caption dict must contain one of: caption, text, prompt")

        raise TypeError(f"Unsupported caption payload type: {type(payload).__name__}")

    def _resolve_caption(self, image_name):
        if image_name in self.captions:
            return self._extract_caption_text(self.captions[image_name])

        image_stem = Path(image_name).stem
        if image_stem in self.captions:
            return self._extract_caption_text(self.captions[image_stem])

        raise KeyError(f"Missing validation caption for {image_name} in {self.caption_json}")

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = self._resolve_caption(image_path.name)
        return image, caption

    def __len__(self):
        return len(self.image_paths)


class CLIPLoss(torch.nn.Module):

    def __init__(self, clip_model_name = "openai/clip-vit-base-patch32"):
        super().__init__()

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).eval()
        self.image_encoder.requires_grad_(False)

        self.transform_for_clip = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2.0),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def forward(self, rec, gt):

        rec_inputs = self.transform_for_clip(rec)
        gt_inputs = self.transform_for_clip(gt)

        rec_features = self.image_encoder(rec_inputs).image_embeds
        gt_features = self.image_encoder(gt_inputs).image_embeds

        rec_features = rec_features / rec_features.norm(p=2, dim=-1, keepdim=True)
        gt_features = gt_features / gt_features.norm(p=2, dim=-1, keepdim=True)

        loss = torch.norm(gt_features - rec_features, p=2, dim=-1).mean()
        return loss
