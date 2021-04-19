# PyTorch models summary

Produce Keras-like summaries of your PyTorch models.

## Example

The following code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_summary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


net = Net()
net.conv2.requires_grad_(False) # make non-trainable
summary(net, input_shape=(1, 28, 28))
```

produces this output:

```text
Model Summary:
╒════╤════════════════════╤═══════════════════════╤═══════════╕
│    │ Layer (type)       │ Output Shape          │ Param #   │
╞════╪════════════════════╪═══════════════════════╪═══════════╡
│  0 │ conv1 (Conv2d)     │ (None, 1, 32, 26, 26) │ 320       │
├────┼────────────────────┼───────────────────────┼───────────┤
│  1 │ conv2 (Conv2d)     │ (None, 1, 64, 24, 24) │ 18,496    │
├────┼────────────────────┼───────────────────────┼───────────┤
│  2 │ dropout1 (Dropout) │ (None, 1, 64, 12, 12) │ 0         │
├────┼────────────────────┼───────────────────────┼───────────┤
│  3 │ fc1 (Linear)       │ (None, 1, 128)        │ 1,179,776 │
├────┼────────────────────┼───────────────────────┼───────────┤
│  4 │ dropout2 (Dropout) │ (None, 1, 128)        │ 0         │
├────┼────────────────────┼───────────────────────┼───────────┤
│  5 │ fc2 (Linear)       │ (None, 1, 10)         │ 1,290     │
╘════╧════════════════════╧═══════════════════════╧═══════════╛
Total params: 1,199,882
Trainable params: 1,181,386
Non-trainable params: 18,496
```
