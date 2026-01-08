"""
Synthetic Video Generation for Edge Cases
GAN-based approach to generate synthetic surveillance video data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class VideoFrameDataset(Dataset):
    """Dataset for loading video frames for GAN training"""
    
    def __init__(self, video_paths, frame_size=(128, 128), max_frames_per_video=100):
        self.frame_size = frame_size
        self.frames = []
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Extract frames from videos
        for video_path in video_paths:
            self.extract_frames_from_video(video_path, max_frames_per_video)
        
        logger.info(f"Loaded {len(self.frames)} frames for training")
    
    def extract_frames_from_video(self, video_path, max_frames):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            
            # Apply transforms
            tensor_frame = self.transform(frame)
            self.frames.append(tensor_frame)
            
            frame_count += 1
            
            # Skip frames for efficiency
            for _ in range(5):  # Skip 5 frames
                cap.read()
        
        cap.release()
        logger.info(f"Extracted {frame_count} frames from {video_path}")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return self.frames[idx]

class Generator(nn.Module):
    """Generator network for creating synthetic video frames"""
    
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.nz = nz
        
        self.main = nn.Sequential(
            # Input: Z goes into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            
            # State size: (ngf//2) x 64 x 64
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            
            # State size: (nc) x 128 x 128
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """Discriminator network for distinguishing real vs synthetic frames"""
    
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class VideoGAN:
    """Main VideoGAN class for training and generating synthetic videos"""
    
    def __init__(self, device='cuda', nz=100, lr=0.0002, beta1=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.nz = nz
        self.lr = lr
        self.beta1 = beta1
        
        # Initialize networks
        self.netG = Generator(nz=nz).to(self.device)
        self.netD = Discriminator().to(self.device)
        
        # Initialize weights
        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Training statistics
        self.G_losses = []
        self.D_losses = []
        
        logger.info(f"VideoGAN initialized on device: {self.device}")
    
    @staticmethod
    def weights_init(m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def train(self, dataloader, num_epochs=100, save_interval=10, output_dir='gan_output'):
        """Train the GAN model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create fixed noise for generating samples during training
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)
        
        # Real and fake labels
        real_label = 1.0
        fake_label = 0.0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ############################
                # Train with all-real batch
                self.netD.zero_grad()
                real_cpu = data.to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                
                output = self.netD(real_cpu)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                
                # Train with all-fake batch
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = self.netG(noise)
                label.fill_(fake_label)
                
                output = self.netD(fake.detach())
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizerD.step()
                
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ############################
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                
                output = self.netD(fake)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()
                
                # Save losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                
                # Output training stats
                if i % 50 == 0:
                    logger.info(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                              f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                              f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            
            # Save generated samples
            if epoch % save_interval == 0:
                with torch.no_grad():
                    fake_samples = self.netG(fixed_noise)
                    vutils.save_image(fake_samples.detach(),
                                    f'{output_dir}/fake_samples_epoch_{epoch:03d}.png',
                                    normalize=True, nrow=8)
                
                # Save model checkpoints
                torch.save(self.netG.state_dict(), f'{output_dir}/generator_epoch_{epoch}.pth')
                torch.save(self.netD.state_dict(), f'{output_dir}/discriminator_epoch_{epoch}.pth')
        
        logger.info("Training completed!")
        
        # Save final models
        torch.save(self.netG.state_dict(), f'{output_dir}/generator_final.pth')
        torch.save(self.netD.state_dict(), f'{output_dir}/discriminator_final.pth')
        
        # Plot training losses
        self.plot_losses(output_dir)
    
    def plot_losses(self, output_dir):
        """Plot training losses"""
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{output_dir}/training_losses.png')
        plt.close()
    
    def load_generator(self, model_path):
        """Load a pre-trained generator model"""
        self.netG.load_state_dict(torch.load(model_path, map_location=self.device))
        self.netG.eval()
        logger.info(f"Generator loaded from {model_path}")
    
    def generate_synthetic_frames(self, num_frames=100, output_dir='synthetic_frames'):
        """Generate synthetic frames using the trained generator"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.netG.eval()
        generated_frames = []
        
        with torch.no_grad():
            for i in range(num_frames):
                # Generate random noise
                noise = torch.randn(1, self.nz, 1, 1, device=self.device)
                
                # Generate frame
                fake_frame = self.netG(noise)
                
                # Convert to numpy array
                frame_np = fake_frame.squeeze().cpu().numpy()
                frame_np = (frame_np + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
                frame_np = np.transpose(frame_np, (1, 2, 0))  # CHW to HWC
                frame_np = (frame_np * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                generated_frames.append(frame_bgr)
                
                # Save individual frame
                cv2.imwrite(f'{output_dir}/frame_{i:04d}.png', frame_bgr)
        
        logger.info(f"Generated {num_frames} synthetic frames")
        return generated_frames
    
    def create_synthetic_video(self, num_frames=100, fps=30, output_path='synthetic_video.avi'):
        """Create a synthetic video from generated frames"""
        frames = self.generate_synthetic_frames(num_frames)
        
        if not frames:
            logger.error("No frames generated")
            return
        
        # Get frame dimensions
        height, width, channels = frames[0].shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames to video
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Synthetic video created: {output_path}")

class SyntheticAnomalyGenerator:
    """Generate synthetic anomalous scenarios for training"""
    
    def __init__(self, video_gan):
        self.video_gan = video_gan
        self.anomaly_types = ['loitering', 'unusual_movement', 'object_abandonment', 'erratic_behavior']
    
    def generate_loitering_scenario(self, duration=60, output_path='loitering_scenario.avi'):
        """Generate a video showing loitering behavior"""
        frames = []
        
        # Generate base scene
        with torch.no_grad():
            # Create a consistent background
            noise_base = torch.randn(1, self.video_gan.nz, 1, 1, device=self.video_gan.device)
            
            for i in range(duration * 30):  # 30 fps
                # Add slight variations to simulate person staying in same area
                noise_variation = noise_base + torch.randn_like(noise_base) * 0.1
                
                frame = self.video_gan.netG(noise_variation)
                frame_np = self.tensor_to_frame(frame)
                frames.append(frame_np)
        
        self.save_video(frames, output_path)
        return output_path
    
    def generate_unusual_movement_scenario(self, duration=30, output_path='unusual_movement_scenario.avi'):
        """Generate a video showing unusual movement patterns"""
        frames = []
        
        with torch.no_grad():
            for i in range(duration * 30):
                # Create erratic noise patterns
                if i % 10 < 5:  # Sudden direction changes
                    noise = torch.randn(1, self.video_gan.nz, 1, 1, device=self.video_gan.device) * 2
                else:
                    noise = torch.randn(1, self.video_gan.nz, 1, 1, device=self.video_gan.device) * 0.5
                
                frame = self.video_gan.netG(noise)
                frame_np = self.tensor_to_frame(frame)
                frames.append(frame_np)
        
        self.save_video(frames, output_path)
        return output_path
    
    def generate_object_abandonment_scenario(self, duration=45, output_path='abandonment_scenario.avi'):
        """Generate a video showing object abandonment"""
        frames = []
        
        with torch.no_grad():
            base_noise = torch.randn(1, self.video_gan.nz, 1, 1, device=self.video_gan.device)
            
            for i in range(duration * 30):
                if i < duration * 15:  # First half: person with object
                    noise = base_noise + torch.randn_like(base_noise) * 0.2
                else:  # Second half: object left behind
                    noise = base_noise + torch.randn_like(base_noise) * 0.05
                
                frame = self.video_gan.netG(noise)
                frame_np = self.tensor_to_frame(frame)
                frames.append(frame_np)
        
        self.save_video(frames, output_path)
        return output_path
    
    def tensor_to_frame(self, tensor):
        """Convert tensor to OpenCV frame"""
        frame_np = tensor.squeeze().cpu().numpy()
        frame_np = (frame_np + 1) / 2.0
        frame_np = np.transpose(frame_np, (1, 2, 0))
        frame_np = (frame_np * 255).astype(np.uint8)
        return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    
    def save_video(self, frames, output_path, fps=30):
        """Save frames as video file"""
        if not frames:
            return
        
        height, width, channels = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Synthetic scenario saved: {output_path}")
    
    def generate_all_scenarios(self, output_dir='synthetic_scenarios'):
        """Generate all types of anomalous scenarios"""
        os.makedirs(output_dir, exist_ok=True)
        
        scenarios = []
        
        # Generate different anomaly scenarios
        scenarios.append(self.generate_loitering_scenario(
            output_path=f'{output_dir}/loitering_scenario.avi'))
        
        scenarios.append(self.generate_unusual_movement_scenario(
            output_path=f'{output_dir}/unusual_movement_scenario.avi'))
        
        scenarios.append(self.generate_object_abandonment_scenario(
            output_path=f'{output_dir}/abandonment_scenario.avi'))
        
        # Generate metadata
        metadata = {
            'generation_time': datetime.now().isoformat(),
            'scenarios': [
                {'type': 'loitering', 'file': 'loitering_scenario.avi', 'duration': 60},
                {'type': 'unusual_movement', 'file': 'unusual_movement_scenario.avi', 'duration': 30},
                {'type': 'object_abandonment', 'file': 'abandonment_scenario.avi', 'duration': 45}
            ]
        }
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated {len(scenarios)} synthetic scenarios in {output_dir}")
        return scenarios

def train_video_gan_on_avenue(avenue_dataset_path, output_dir='video_gan_output'):
    """Train VideoGAN on Avenue dataset"""
    # Find all video files
    video_paths = list(Path(avenue_dataset_path).glob("*.avi"))
    
    if not video_paths:
        logger.error(f"No video files found in {avenue_dataset_path}")
        return None
    
    logger.info(f"Found {len(video_paths)} video files")
    
    # Create dataset
    dataset = VideoFrameDataset(video_paths, frame_size=(128, 128))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # Initialize and train GAN
    video_gan = VideoGAN()
    video_gan.train(dataloader, num_epochs=200, output_dir=output_dir)
    
    return video_gan

def generate_synthetic_scenarios(generator_path, output_dir='synthetic_output'):
    """Generate synthetic scenarios using trained generator"""
    # Initialize VideoGAN and load generator
    video_gan = VideoGAN()
    video_gan.load_generator(generator_path)
    
    # Create synthetic anomaly generator
    anomaly_gen = SyntheticAnomalyGenerator(video_gan)
    
    # Generate all scenarios
    scenarios = anomaly_gen.generate_all_scenarios(output_dir)
    
    return scenarios

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Synthetic Video Generation for Surveillance')
    parser.add_argument('--mode', choices=['train', 'generate'], required=True,
                       help='Mode: train GAN or generate synthetic videos')
    parser.add_argument('--avenue_path', type=str, 
                       help='Path to Avenue dataset for training')
    parser.add_argument('--generator_path', type=str,
                       help='Path to trained generator model')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.avenue_path:
            print("Please provide Avenue dataset path with --avenue_path")
            exit(1)
        
        video_gan = train_video_gan_on_avenue(args.avenue_path, args.output_dir)
        if video_gan:
            print(f"Training completed. Models saved in {args.output_dir}")
    
    elif args.mode == 'generate':
        if not args.generator_path:
            print("Please provide generator model path with --generator_path")
            exit(1)
        
        scenarios = generate_synthetic_scenarios(args.generator_path, args.output_dir)
        print(f"Generated {len(scenarios)} synthetic scenarios in {args.output_dir}")
    
    else:
        print("Invalid mode selected")