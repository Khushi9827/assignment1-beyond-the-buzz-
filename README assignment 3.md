# byb_ass3
import torch
from torch.utils.data import Dataset

class DinosaurDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = file.read().lower().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]





import string

def one_hot_encode(char, char_to_index):
    vector = torch.zeros(len(char_to_index))
    index = char_to_index[char]
    vector[index] = 1.0
    return vector

def decode_one_hot(vector, index_to_char):
    _, max_index = vector.max(dim=0)
    char = index_to_char[max_index.item()]
    return char

def create_char_mappings(data):
    chars = sorted(list(set(''.join(data))))
    char_to_index = {char: index for index, char in enumerate(chars)}
    index_to_char = {index: char for index, char in enumerate(chars)}
    return char_to_index, index_to_char




import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output



import torch.optim as optim

# Define hyperparameters
input_size = len(char_to_index)
hidden_size = 128
output_size = len(char_to_index)
num_epochs = 100
learning_rate = 0.01

# Create RNN model
model = RNN(input_size, hidden_size, output_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for name in data:
        name = name + '\n'  # Add end-of-line character
        name_encoded = [one_hot_encode(c, char_to_index) for c in name]
        name_input = torch.stack(name_encoded[:-1]).unsqueeze(0)
        name_target = torch.tensor([char_to_index[c] for c in name[1:]])

        optimizer.zero_grad()

        output = model(name_input)
        output = output.view(-1, output_size)
        loss = criterion(output, name_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(data)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')




# Generate dinosaur names
with torch.no_grad():
    random_input = torch.randn(1, 1, input_size)
    hidden = torch.zeros(1, 1, hidden_size)
    generated_name = []
    
    while True:
        output, hidden = model(random_input, hidden)
        output = output.squeeze(0)
        char_index = output.argmax().item()
        char = index_to_char[char_index]
        if char == '\n':
            break
        generated_name.append(char)
        random_input = one_hot_encode(char, char_to_index).unsqueeze(0).unsqueeze(0)

    generated_name = ''.join(generated_name)
    print(f'Generated Name: {generated_name}')
