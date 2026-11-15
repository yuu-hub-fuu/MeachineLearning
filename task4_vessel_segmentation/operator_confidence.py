"""Map native confidence through the exact binary submission operator.

Layer-cake quadrature of B(p>=t), t in a fixed uniform grid. This yields a
monotone soft confidence map for comparing/fusing native and 512 predictors;
it is not claimed to be a calibrated clinical probability.
"""
import numpy as np
import v3_native as native


def lift(p,levels=100):
    result=[]
    thresholds=(np.arange(levels,dtype=np.float32)+.5)/levels
    for a in p:
        count=np.zeros((512,512),np.uint16)
        for t in thresholds:
            count+=native.convert_masks(a[None],float(t))[0].astype(np.uint16)
        result.append(count.astype(np.float32)/levels)
    return np.stack(result)
