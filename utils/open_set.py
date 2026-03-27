import torch


def calibrate_unknown_score(unknown_score):
    return torch.clamp(unknown_score, min=0.0, max=1.0)


def fuse_unknown_score(mass_unknown_score, distance_unknown_score, alpha=0.5):
    mass_unknown_score = calibrate_unknown_score(mass_unknown_score)
    distance_unknown_score = calibrate_unknown_score(distance_unknown_score)
    return calibrate_unknown_score(alpha * mass_unknown_score + (1.0 - alpha) * distance_unknown_score)


def fit_threshold(scores, method='quantile', q=0.95):
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    scores = scores.float().view(-1)
    if scores.numel() == 0:
        return torch.tensor(0.5)

    if method == 'mean_std':
        return scores.mean() + scores.std(unbiased=False)
    return torch.quantile(scores, q)


def predict_open_set(class_scores, unknown_score, threshold, unknown_index):
    unknown_score = calibrate_unknown_score(unknown_score)
    prediction = class_scores.argmax(dim=1)
    prediction = prediction.clone()
    prediction[unknown_score >= threshold] = unknown_index
    return prediction
