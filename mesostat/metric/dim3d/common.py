import numpy as np


# Shuffle array elements of arbitrary dimension
def _shuffle_ND(data):
    data1D = data.flatten()
    np.random.shuffle(data1D)
    return data1D.reshape(data.shape)


# Shuffle samples and repetitions for target channel only
def shuffle_target(dataRPS: np.array, trg: int, settings: dict):
    if 'shuffle' in settings and settings['shuffle']:
        dataEff = np.copy(dataRPS)
        dataEff[:, trg] = _shuffle_ND(dataEff[:, trg])
        return dataEff
    else:
        return dataRPS


# Extract channel indices from settings dictionary
def parse_channels(settings: dict, dim: int):
    if 'channels' in settings.keys():
        assert len(settings['channels']) == dim
        src = settings['channels'][:-1]
        trg = settings['channels'][-1]
    else:
        src = settings['src']
        trg = settings['trg']
    return [int(s) for s in src], int(trg)