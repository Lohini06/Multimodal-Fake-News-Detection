import torch
import random
import numpy as np
from src.utils.config import load_config, ensure_dirs
from src.data.preprocess import create_sample_dataset, load_and_validate
from src.data.dataset import create_dataloaders
from src.models.clip_classifier import MultimodalClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    evaluate, compute_metrics, save_metrics,
    plot_confusion_matrix, plot_training_history
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    config = load_config('config.yaml')
    ensure_dirs(config)
    set_seed(config['training']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))

    print('Preparing dataset...')
    df = create_sample_dataset(config['data']['processed_dir'])
    df = load_and_validate(config['data']['processed_dir'] + '/sample_data.csv')

    print('Loading CLIP model...')
    model = MultimodalClassifier(config)
    print('Model ready.')

    print('Creating dataloaders...')
    train_loader, val_loader, test_loader = create_dataloaders(
        df, model.preprocess, config
    )

    trainer = Trainer(model, config, device)
    history = trainer.fit(train_loader, val_loader)

    print('Evaluating on test set...')
    preds, labels, probs = evaluate(model, test_loader, device, config)
    metrics = compute_metrics(preds, labels)
    save_metrics(metrics, config)

    print('Saving plots...')
    plot_confusion_matrix(preds, labels, config)
    plot_training_history(history, config)

    print('Pipeline complete! Check outputs/ for results.')

if __name__ == '__main__':
    main()