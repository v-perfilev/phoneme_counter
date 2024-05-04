import torch
from torch.nn.utils.rnn import pad_sequence


def device_collate_fn(batch, use_cuda=False, use_mps=False):
    # Determine the device based on availability and user preference
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Prepare spectrograms and syllable counts
    spectrograms, syllable_counts = zip(*batch)
    spectrograms = [s.squeeze(0).permute(1, 0) for s in spectrograms]
    spectrograms = pad_sequence(spectrograms, batch_first=True)
    syllable_counts = torch.tensor(syllable_counts)

    # Move spectrogram tensor in the batch to the selected device and convert to float32
    spectrograms = [x.to(device).to(torch.float32) for x in spectrograms]

    # Move syllable count tensor in the batch to the selected device and convert to float32
    syllable_counts = [x.to(device).to(torch.float32) for x in syllable_counts]

    return spectrograms, syllable_counts


def to_device_fn(obj, use_cuda=False, use_mps=False):
    # Determine the device based on availability and user preference
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Move the object to the selected device and to float32 data type
    return obj.to(device).to(torch.float32)
