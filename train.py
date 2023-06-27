import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define transform for normalizing the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)




import torch.optim as optim

# Define hyperparameters
latent_dim = 100
image_dim = 28 * 28

# Create generator and discriminator
generator = Generator(latent_dim, image_dim)
discriminator = Discriminator(image_dim)

# Define optimizers
lr = 0.0002
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Define loss criterion
criterion = nn.BCELoss()





num_epochs = 100

for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)

        # Flatten the images
        real_images = real_images.view(batch_size, -1)

        # Train discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Forward pass real images through discriminator
        real_output = discriminator(real_images)
        real_loss = criterion(real_output, real_labels)
        real_score = torch.mean(real_output).item()

        # Generate fake images
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)

        # Forward pass fake images through discriminator
        fake_output = discriminator(fake_images)
        fake_loss = criterion(fake_output, fake_labels)
        fake_score = torch.mean(fake_output).item()

        # Compute total discriminator loss and backpropagate
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train generator
        generator.zero_grad()
        labels = torch.ones(batch_size, 1)

        # Forward pass fake images through discriminator again
        output = discriminator(fake_images)
        generator_loss = criterion(output, labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Print progress
        if (batch_idx + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Discriminator Loss: {discriminator_loss.item():.4f}, '
                  f'Generator Loss: {generator_loss.item():.4f}, '
                  f'Real Score: {real_score:.4f}, Fake Score: {fake_score:.4f}')

    # Save generated images
    with torch.no_grad():
        fake_images = generator(torch.randn(10, latent_dim))
        fake_images = fake_images.view(-1, 1, 28, 28)
        torchvision.utils.save_image(fake_images, f'generated_images_epoch_{epoch+1}.png', nrow=10, normalize=True)
