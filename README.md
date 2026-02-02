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
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x


ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()


        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```

### Dataset Information


<img width="212" height="396" alt="image" src="https://github.com/user-attachments/assets/5be43ea2-bf77-4b8f-a0b8-a88ee0923e89" />


### OUTPUT

### Training Loss Vs Iteration Plot


<img width="771" height="571" alt="image" src="https://github.com/user-attachments/assets/66f9ea74-58ff-421f-b865-0e08def5c91c" />


### New Sample Data Prediction


<img width="920" height="129" alt="image" src="https://github.com/user-attachments/assets/b39d667f-18f2-4fcf-9241-9252c589513e" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
