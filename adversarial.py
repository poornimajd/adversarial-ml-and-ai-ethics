import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from PIL import Image
import torchvision.utils as vutils
if torch.backends.mps.is_available():
	device=torch.device("mps")
elif torch.cuda.is_available():
	device=torch.device("cuda")
else:
	device=torch.device("cpu")

print(device)

# define CNN for a 3-class problem with input size 160x160 images
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(256 * 5 * 5, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 3)
		self.relu = nn.ReLU()
		self.final_activation = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = self.pool(self.relu(self.conv3(x)))
		x = self.pool(self.relu(self.conv4(x)))
		x = self.pool(self.relu(self.conv5(x)))
		x = x.view(-1, 256 * 5 * 5)
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		x = self.final_activation(x)
		return x



# Load dataset
train_dir = '/content/drive/My Drive/Colab Notebooks/fai/data/train'
test_dir = '/content/drive/My Drive/Colab Notebooks/fai/data/test'
image_size = 160
batch_size = 16
workers = 0

class CropToSmallerDimension(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img):
        # Get the original image size
        width, height = img.size

        # Determine the smaller dimension
        smaller_dimension = min(width, height)

        # Crop the image to the smaller dimension
        return transforms.CenterCrop(smaller_dimension)(img)

train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms.Compose([CropToSmallerDimension(256),transforms.ToTensor(),transforms.Resize(image_size)]))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms.Compose([CropToSmallerDimension(256),transforms.ToTensor(),transforms.Resize(image_size)]))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

print('Number of training images: {}'.format(len(train_dataset)))
print('Number of test images: {}'.format(len(test_dataset)))
print('Detected Classes are: ', train_dataset.classes) # classes are detected by folder structure

# Define the attack
def FGSM(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image







net = Net()
net.to(device)

# Train the network

# criterion = nn.NLLLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# epochs = 100
# running_loss = 0
# train_losses, test_losses = [], []
# i=0

# for epoch in tqdm(range(epochs)):
# 	for inputs, labels in train_dataloader:
# 		inputs, labels = inputs.to(device), labels.to(device)
# 		optimizer.zero_grad()
# 		logps = net(inputs)
# 		loss = criterion(logps, labels)
# 		loss.backward()
# 		optimizer.step()
# 		running_loss += loss.item()

# # Save the model
# torch.save(net.state_dict(), 'model.pth')


# Test the model
net.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/fai/model.pth', map_location="cpu"))
net.to(device)

correct=[]

net.eval()
accuracy = 0
for inputs, labels in tqdm(test_dataloader):
	inputs, labels = inputs.to(device), labels.to(device)
	outputs = net(inputs)
	_, predicted = torch.max(outputs.data, 1)
	accuracy += (predicted == labels).sum().item()
	correct.append((predicted == labels).tolist())

print('Accuracy of the network on the test images: %d %%' % (100 * accuracy / len(test_dataset)))


# Test the model with adversarial examples
# Save adversarial examples for each class using FGSM with (eps = 0.01, 0.05, 0.1)
# Save one adversarial example for each class using PGD with (eps = 0.01, 0.05, 0.1, alpha = 0.001, 0.005, 0.01 respectively, iterations = 20)
criterion = nn.NLLLoss()

# Function to test the model with adversarial examples
def test_model_with_adversarial(net, test_loader, epsilon):
    correct = 0
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        inputs.requires_grad = True

        # Forward pass the data through the model
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Zero all existing gradients
        net.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = inputs.grad.data

        # Call FGSM Attack
        perturbed_data = FGSM(inputs, epsilon, data_grad)

        # Re-classify the perturbed image
        outputs = net(perturbed_data)
        _, final_pred = torch.max(outputs.data, 1)

        correct += (final_pred == labels).sum().item()

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader.dataset))*100
    print('Epsilon:', epsilon, 'Test Accuracy =(%d / %d)*100 = %f' % (correct, len(test_loader.dataset), final_acc))

    return final_acc

# Evaluate the network for each alpha
for alpha in [0.001, 0.01, 0.1]:
    test_model_with_adversarial(net, test_dataloader, alpha)


# Dictionary to store one example per class
selected_examples = {'apple': None, 'banana': None, 'orange': None}

# Iterate through test dataset to select one example per class
for inputs, labels in test_dataloader:
    for i, label in enumerate(labels):
        class_name = train_dataset.classes[label]
        if class_name in selected_examples and selected_examples[class_name] is None:
            selected_examples[class_name] = inputs[i].unsqueeze(0)  # Add batch dimension
            if all(example is not None for example in selected_examples.values()):
                break  # Break if all classes have an example

# Check if all classes are selected
assert all(example is not None for example in selected_examples.values()), "Not all classes have an example selected"




def save_image(tensor, filename):
    """
    Saves a tensor as an image.

    Args:
    tensor (Tensor): The image tensor to save.
    filename (str): The path to the file in which the image will be saved.
    """
    # Convert the tensor to an image and save it
    vutils.save_image(tensor, filename)



# Function to create and save adversarial examples




def add_text_to_image(tensor, text):
    # Convert tensor to PIL Image
    pil_img = transforms.ToPILImage()(tensor.cpu())

    # Create a new image with extra space at the top for text
    new_height = pil_img.height + 30  # Adjust the 30 pixels as needed
    new_img = Image.new("RGB", (pil_img.width, new_height), (255, 255, 255))
    new_img.paste(pil_img, (0, 30))

    # Create a drawing object
    draw = ImageDraw.Draw(new_img)

    # Use a default PIL font
    font = ImageFont.load_default()

    # Add text to the new image
    draw.text((10, 5), text, font=font, fill='black')  # Position the text at the top

    return new_img

def create_and_save_adversarial_examples(net, examples, alphas, folder='/content/drive/My Drive/Colab Notebooks/fai/adversarial_examples'):
    os.makedirs(folder, exist_ok=True)
    for class_name, input_tensor in examples.items():
        input_tensor = input_tensor.to(device)
        original_output = net(input_tensor)
        original_pred = torch.max(original_output, 1)[1]

        for alpha in alphas:
            # Generate adversarial example
            input_tensor.requires_grad = True
            output = net(input_tensor)
            loss = F.nll_loss(output, original_pred)
            net.zero_grad()
            loss.backward()
            data_grad = input_tensor.grad.data
            perturbed_data = FGSM(input_tensor, alpha, data_grad)

            # Get prediction for adversarial example
            perturbed_output = net(perturbed_data)
            perturbed_pred = torch.max(perturbed_output, 1)[1]
            predicted_class = train_dataset.classes[perturbed_pred.item()]
            text = f"Alpha: {alpha}, Pred: {predicted_class}"
            annotated_img = add_text_to_image(perturbed_data.squeeze(), text)

            # Save the adversarial example with annotation
            filename = f"{folder}/{class_name}_alpha_{alpha}_pred_{predicted_class}.png"
            annotated_img.save(filename)

# Create and save adversarial examples
create_and_save_adversarial_examples(net, selected_examples, [0.001, 0.01, 0.1])


def PGD(model, image, label, epsilon, alpha, iterations):
    # Start with a copy of the original image
    perturbed_image = image.clone().detach().to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    perturbed_image.requires_grad = True
    model.eval()
    for _ in range(iterations):
        # Forward pass the perturbed image through the model
        model.zero_grad()
        output = model(perturbed_image)
        loss = F.nll_loss(output, label)
        # Zero all existing gradients
        
        # Calculate gradients of model in backward pass
        loss.backward()
        with torch.no_grad():
            perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
            perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1) 

        perturbed_image.requires_grad_(True)

    return perturbed_image.detach()
    

def test_model_with_pgd(net, test_loader, epsilon, alpha, iterations):
    correct = 0
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Generate adversarial example using PGD
        perturbed_data = PGD(net, inputs, labels, epsilon, alpha, iterations)

        # Re-classify the perturbed image
        outputs = net(perturbed_data)
        _, final_pred = torch.max(outputs.data, 1)

        correct += (final_pred == labels).sum().item()

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader.dataset))
    print('Epsilon:', epsilon, 'Test Accuracy = %d / %d = %f' % (correct, len(test_loader.dataset), final_acc))

    return final_acc

# Constants
alpha = 2/255
iterations = 50

# Evaluate the network for each epsilon
for epsilon in [0.01, 0.05, 0.1]:
    test_model_with_pgd(net, test_dataloader, epsilon, alpha, iterations)




def add_text_to_image(tensor, text):
    # Convert tensor to PIL Image
    pil_img = transforms.ToPILImage()(tensor.cpu())

    # Create a new image with extra space at the top for text
    new_height = pil_img.height + 30  # Adjust the 30 pixels as needed
    new_img = Image.new("RGB", (pil_img.width, new_height), (255, 255, 255))
    new_img.paste(pil_img, (0, 30))

    # Create a drawing object
    draw = ImageDraw.Draw(new_img)

    # Use a default PIL font
    font = ImageFont.load_default()

    # Add text to the new image
    draw.text((10, 5), text, font=font, fill='black')  # Position the text at the top

    return new_img

def save_image_with_text(tensor, filename, text):
    # Remove the batch dimension (if present) by squeezing the tensor
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Squeeze the tensor to remove the batch dimension

    annotated_img = add_text_to_image(tensor, text)
    annotated_img.save(filename)

# Directory to save images
save_dir = '/content/drive/My Drive/Colab Notebooks/fai/adversarial_examples_pgd'
os.makedirs(save_dir, exist_ok=True)

# Dictionary to store one example per class
selected_examples = {'apple': None, 'banana': None, 'orange': None}

# Iterate through test dataset to select one example per class
for inputs, labels in test_dataloader:
    for i, label in enumerate(labels):
        class_name = train_dataset.classes[label]
        if class_name in selected_examples and selected_examples[class_name] is None:
            selected_examples[class_name] = (inputs[i], label)
            if all(selected_examples.values()):
                break

# PGD parameters
alpha = 2/255
iterations = 50
epsilons = [0.01, 0.05, 0.1]

# Generate and save adversarial examples
for class_name, (input_tensor, label) in selected_examples.items():
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
    label = label.unsqueeze(0).to(device)

    # Save original image
    original_filename = os.path.join(save_dir, f'{class_name}_original.png')
    save_image_with_text(input_tensor, original_filename, f"Original - {class_name}")

    # Generate and save adversarial examples for each epsilon
    for epsilon in epsilons:
        perturbed_data = PGD(net, input_tensor, label, epsilon, alpha, iterations)
        perturbed_output = net(perturbed_data)
        predicted_class = train_dataset.classes[torch.max(perturbed_output, 1)[1]]

        # Add text annotation and save adversarial example
        text = f"Epsi: {epsilon}, Pred: {predicted_class}"
        adv_filename = os.path.join(save_dir, f'{class_name}_adv_epsilon_{epsilon}_pred_{predicted_class}.png')
        save_image_with_text(perturbed_data.squeeze(), adv_filename, text)




# Constants for PGD
epsilon = 0.075
alpha = 2/255
iterations = 50

# Load the pre-trained model
net.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/fai/model.pth', map_location=device))
net.to(device)

# Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.NLLLoss()  # Assuming you are still using negative log-likelihood loss

# Training loop
for epoch in range(10):  # 10 new epochs
    running_loss = 0.0
    for inputs, labels in tqdm(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Generate adversarial examples for the current batch
        inputs.requires_grad = True
        adversarial_inputs = PGD(net, inputs, labels, epsilon, alpha, iterations)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(adversarial_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}')

# Optionally, save the updated model
torch.save(net.state_dict(), '/content/drive/My Drive/Colab Notebooks/fai/updated_model.pth')



def evaluate_model_on_clean_data(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy on clean test images: %f %%' % accuracy)
    return accuracy

clean_accuracy = evaluate_model_on_clean_data(net, test_dataloader)


def evaluate_model_on_adversarial_data(model, test_loader, epsilon, alpha, iterations):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        perturbed_data = PGD(model, inputs, labels, epsilon, alpha, iterations)
        outputs = model(perturbed_data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy on adversarial test images: %f %%' % accuracy)
    return accuracy

# Parameters for PGD
epsilon = 0.075
alpha = 2/255
iterations = 50

adv_accuracy = evaluate_model_on_adversarial_data(net, test_dataloader, epsilon, alpha, iterations)
