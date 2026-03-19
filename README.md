# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model

<img width="1276" height="826" alt="Screenshot 2026-02-02 094505" src="https://github.com/user-attachments/assets/f887f0ca-cd2f-4815-8f7e-f4d7c9e43ac3" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM


### Name:MANIKANDAN T

### Register Number:212224110037

```python
# Name:MANIKANDAN T
# Register Number:212224110037
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'Loss':[]}
        
  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
   for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(ai_brain(X_train),y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['Loss'].append(loss.item()) # Corrected: 'loss' changed to 'Loss'
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_n1_1 = torch.tensor([[13]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')



```

### Dataset Information


<img width="212" height="396" alt="image" src="https://github.com/user-attachments/assets/5be43ea2-bf77-4b8f-a0b8-a88ee0923e89" />


### OUTPUT

### Training Loss Vs Iteration Plot


<img width="844" height="639" alt="image" src="https://github.com/user-attachments/assets/0b4e8ee3-7daa-4d30-8d5e-56bccfe9cac9" />


### New Sample Data Prediction


<img width="389" height="168" alt="image" src="https://github.com/user-attachments/assets/3e2ba500-5cb0-4952-89a0-48a19b98fb37" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
