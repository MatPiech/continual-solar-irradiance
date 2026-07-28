from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from continual_solar_irradiance.models.forecaster import Forecaster


IMG_MEAN = [ 0.485, 0.456, 0.406 ]
IMG_STD = [ 0.229, 0.224, 0.225 ]

CROP_MASK = cv2.circle(
    np.zeros((128, 128, 3), dtype=np.uint8),
    (128 // 2, 128 // 2),
    128 // 2,
    color=(1, 1, 1),
    thickness=-1,
)


def preprocess_input(period):
    image_path = period['history'][-1]['image_name']
    irradiance_history = [item['irradiance'] for item in period['history']]
    target_irradiance = period['target_irradiance']

    img = np.asarray(Image.open(image_path).convert('RGB'))[:, 254:-254]
    img = cv2.resize(img, (128, 128)) / 255.0
    img = (img - IMG_MEAN ) / IMG_STD
    img = np.where(CROP_MASK, img, 0.).astype(np.float32)

    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    irradiance_history_tensor = torch.tensor(irradiance_history, dtype=torch.float32)
    target_irradiance = torch.tensor(target_irradiance, dtype=torch.float32)

    return img_tensor, irradiance_history_tensor, target_irradiance


@click.command()
@click.option('--checkpoint-path', type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), required=True)
@click.option('--periods-path', type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), required=True)
def compute_embeddings(checkpoint_path: Path, periods_path: Path):
    model = Forecaster.load_from_checkpoint(checkpoint_path)

    with periods_path.open('rb') as f:
        periods = pd.read_pickle(f)

    image_encoder_embeddings_output = []
    latent_space_embeddings_output = []

    for period in tqdm(periods):
        measurement_hour = int(Path(period['history'][-1]['image_name']).name[12:14])
        if measurement_hour < 6 or measurement_hour >= 22:
            continue
        img_tensor, irradiance_history_tensor, _ = preprocess_input(period)

        image_encoder_embeddings = model.image_encoder(img_tensor.unsqueeze(0))
        image_encoder_embeddings_output.append(image_encoder_embeddings.squeeze().detach().numpy())
        latent_space_embeddings_output.append(
            torch.cat([image_encoder_embeddings, irradiance_history_tensor.unsqueeze(0)], dim=1).squeeze().detach().numpy()
        )

    print(f'Image encoder embeddings shape: {np.stack(image_encoder_embeddings_output).shape}')
    print(f'Latent space embeddings shape: {np.stack(latent_space_embeddings_output).shape}')
    np.save(periods_path.parent / f'{periods_path.parents[0].name.lower()}_image_encoder_embeddings_1.npy', np.stack(image_encoder_embeddings_output))
    np.save(periods_path.parent / f'{periods_path.parents[0].name.lower()}_encoder_embeddings_1.npy', np.stack(latent_space_embeddings_output))


if __name__ == '__main__':
    compute_embeddings()
