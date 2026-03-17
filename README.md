# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset



## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.

### STEP 3: 

Define the RNN model architecture.

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 

Train the model with loss tracking.

### STEP 6: 

Predict on test data, plot actual vs. predicted prices.





## PROGRAM

### Name: Mohammad Suhael

### Register Number: 212224230164

```python
## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary
# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

## Step 3: Train the Model

def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
  train_losses = []
  model.train()

  for epoch in range(num_epochs):
    total_loss = 0
    for x_batch,y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    train_losses.append(total_loss/ len(train_loader))
    print(f'Epoch [{epoch+1}/{num_epochs}], loss: {total_loss / len(train_loader):.4f}')
  return train_losses

# Plot training loss
print('Name: Mohammad Suhael')
print('Register Number: 212224230164')

train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=20) 

plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name: Mohammad SUhael')
print('Register Number: 212224230164')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')




```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/dd899ade-73ce-4d33-ba3a-a723ce286942" />

## True Stock Price, Predicted Stock Price vs time

<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/4f8c2d9c-4b04-4f55-999e-0fde9efdf3c9" />

### Predictions
<img width="317" height="66" alt="image" src="https://github.com/user-attachments/assets/7b4db9a4-d54d-4f22-b04f-a574d143ecda" />

## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.

