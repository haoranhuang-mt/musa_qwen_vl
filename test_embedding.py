import torch
import torch_musa
from torch.nn import CrossEntropyLoss

loss_fct = CrossEntropyLoss()
label1 = torch.LongTensor([1,2,3,20]).to('musa')
logits = torch.rand([4,20]).to('musa')
loss = loss_fct(logits, label1)
print(loss)

torch.musa.empty_cache()
# weight = torch.load('./weight.0.pt')
# index = torch.load('./index.0.pt')
# input1 = torch.load('/home/dist/ds/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/input.pt.0')
# print(weight, index.tolist())
# print(input1)
