import numpy as np


def hampel_filter_stream(data, window_size=5, threshold=3.0):
    output = []
    buf = []
    for x in data:
        buf.append(x)
        if len(buf) > window_size:
            buf.pop(0)
        if len(buf) < window_size:
            output.append(x)
            continue
        median = np.median(buf)
        mad = np.median(np.abs(np.array(buf) - median)) * 1.4826
        if mad == 0 or abs(x - median) <= threshold * mad:
            output.append(x)
        else:
            output.append(median)
    return output
