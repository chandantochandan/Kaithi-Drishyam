"""Synthetic data generation for Kaithi OCR training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class SyntheticSample:
    """Generated training sample metadata."""

    image_path: Path
    text: str
    split: str


class SyntheticDataGenerator:
    """Create degraded line images from text and a Kaithi-compatible font."""

    def __init__(self, font_path: Path, output_dir: Path, image_size: tuple[int, int] = (512, 96)) -> None:
        self.font_path = font_path
        self.output_dir = output_dir
        self.image_size = image_size
        self.font = ImageFont.truetype(str(font_path), size=42)

    def generate_dataset(
        self,
        texts: list[str],
        samples_per_text: int = 1,
        seed: int = 42,
    ) -> list[SyntheticSample]:
        """Generate train/validation/test line images and manifest."""
        random.seed(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        samples: list[SyntheticSample] = []

        expanded = [(text, idx) for text in texts for idx in range(samples_per_text)]
        random.shuffle(expanded)

        for index, (text, variant) in enumerate(expanded):
            split = self._split_for_index(index, len(expanded))
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            image = self.generate_line_image(text)
            image_path = split_dir / f"sample_{index:06d}_{variant}.png"
            cv2.imwrite(str(image_path), image)
            samples.append(SyntheticSample(image_path=image_path, text=text, split=split))

        manifest = [
            {"image_path": str(sample.image_path), "text": sample.text, "split": sample.split}
            for sample in samples
        ]
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return samples

    def generate_line_image(self, text: str) -> np.ndarray:
        """Render one degraded synthetic OCR line."""
        width, height = self.image_size
        image = Image.new("L", (width, height), color=242)
        draw = ImageDraw.Draw(image)
        draw.text((24, 22), text, fill=38, font=self.font)

        array = np.array(image)
        array = self._apply_aging_effects(array)
        return array

    def _apply_aging_effects(self, image: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 8, image.shape).astype(np.int16)
        aged = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        blur_kernel = random.choice([1, 3])
        if blur_kernel > 1:
            aged = cv2.GaussianBlur(aged, (blur_kernel, blur_kernel), 0)
        stains = np.random.random(image.shape) < 0.003
        aged[stains] = np.random.randint(120, 190)
        return aged

    def _split_for_index(self, index: int, total: int) -> str:
        fraction = index / max(1, total)
        if fraction < 0.8:
            return "train"
        if fraction < 0.9:
            return "validation"
        return "test"
