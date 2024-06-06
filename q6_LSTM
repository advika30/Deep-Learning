import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
data=torch.randint(1,1000,(100,10))
label=torch.randint(0,2,(100,))
dataset=TensorDataset(data,label)
loader=DataLoader(dataset,batch_size=10,shuffle=True)
class Lstmclass(nn.Module):
    def __init__(self,vocab,emb,hidden,out):
        super(Lstmclass,self).__init__()
        self.embedding=nn.Embedding(vocab,emb)
        self.lstm=nn.LSTM(emb,hidden,batch_first=True)
        self.fc=nn.Linear(hidden,out)
        
    def forward(self,x):
         x=self.embedding(x)
         _ ,(hidden, _)=self.lstm(x)
         return self.fc(hidden.squeeze(0))
model=Lstmclass(1000,50,100,2)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())
for epoch in range(10):
   for data,target in loader:
         optimizer.zero_grad()
         output=model(data)
         loss=criterion(output,target)
         loss.backward()
         optimizer.step()
   print(f'epoch:{epoch+1},loss:{loss.item()}') #,
