def action_for(probability, review_threshold, decline_threshold):
    if not 0 <= review_threshold <= 1 or (decline_threshold is not None and not review_threshold < decline_threshold <= 1):
        raise ValueError("Thresholds must be ordered probabilities")
    if decline_threshold is not None and probability >= decline_threshold:
        return "decline"
    if probability >= review_threshold:
        return "review"
    return "approve"
