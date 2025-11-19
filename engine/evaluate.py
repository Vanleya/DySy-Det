import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score

def evaluate(model, dataloader, device, t, ensemble_size, prompt, beta):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for image_tensor, clip_img_tensor, labels in tqdm(dataloader, desc="Testing", unit="batch"):
            image_tensor = image_tensor.to(device)
            clip_img_tensor = clip_img_tensor.to(device)
            labels = labels.to(device)

            outputs = model(image_tensor, clip_img_tensor, ensemble_size, t, prompt)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)


    acc = accuracy_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_probs)
    return acc, ap