# A Implementation of batch normalization LSTM in pytorch

Frok from sysuNie
Modified to be compatible with Pytorch 1.0.0

# To use:

```sh
import torch
import torch.nn as nn
from batch_normalization_LSTM import BNLSTMCell, LSTM


model = LSTM(cell_class=BNLSTMCell, input_size=28, hidden_size=512, batch_first=True, max_length=152)

if __name__ == "__main__":
    size = 28
    dummy1 = torch.rand(300, 2, size)
    out = model(dummy1)
    print(model)
    print(out[0])
```
