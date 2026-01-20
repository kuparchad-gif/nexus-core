import numpy as np


def softmax(x: np.ndarray, axis=1) -> np.ndarray:
    """
    Computes softmax array along the specified axis.
    """
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def calibrate_sentiment_score(
    sentiment: float,
    thresh_neg: float,
    thresh_pos: float,
    zero: float = 0,
) -> float:
    if thresh_neg != (zero - 1) / 2:
        alpha_neg = -(3 * zero - 1 - 4 * thresh_neg) / (2 * zero - 2 - 4 * thresh_neg) / 2
        if -1 < alpha_neg and alpha_neg < 0:
            raise ValueError(f"Incorrect value: {thresh_neg=} is too far from -0.5!")
    if thresh_pos != (zero + 1) / 2:
        alpha_pos = -(4 * thresh_pos - 1 - 3 * zero) / (2 + 2 * zero - 4 * thresh_pos) / 2
        if 0 < alpha_pos and alpha_pos < 1:
            raise ValueError(f"Incorrect value: {thresh_pos=} is too far from 0.5!")
    if sentiment < 0:
        return (2 * zero - 2 - 4 * thresh_neg) * sentiment**2 + (3 * zero - 1 - 4 * thresh_neg) * sentiment + zero
    elif sentiment > 0:
        return (2 + 2 * zero - 4 * thresh_pos) * sentiment**2 + (4 * thresh_pos - 1 - 3 * zero) * sentiment + zero
    return zero


def calibrate_sentiment(
    sentiments: np.ndarray[float],
    thresh_neg: float,
    thresh_pos: float,
    zero: float,
) -> np.ndarray[np.float64]:
    result = np.array(
        [
            calibrate_sentiment_score(sentiment, thresh_neg=thresh_neg, thresh_pos=thresh_pos, zero=zero)
            for sentiment in sentiments
        ]
    )
    return result.astype(np.float64)


def scale_value(value, in_min, in_max, out_min, out_max):
    if in_min <= value <= in_max:
        scaled_value = (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
        return scaled_value.round(3)
    else:
        raise ValueError(f"Input value must be in the range [{in_min}, {in_max}]")



def get_sentiment(
    logits: np.ndarray,
    thresh_neg: float,
    thresh_pos: float,
    zero: float,
):
    probabilities = softmax(logits, axis=1)
    sentiments = np.matmul(probabilities, np.arange(5)) / 2 - 1
    score = calibrate_sentiment(
        sentiments=sentiments,
        thresh_neg=thresh_neg,
        thresh_pos=thresh_pos,
        zero=zero,
    )[0]
    if score < -0.33:
        return scale_value(score, -1, -0.33, 0, 1), "NEGATIVE"
    elif score < 0.33:
        return scale_value(score, -0.33, 0.33, 0, 1), "NEUTRAL"
    else:
        return scale_value(score, 0.33, 1, 0, 1), "POSITIVE"
