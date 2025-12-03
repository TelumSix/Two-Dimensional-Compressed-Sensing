"""
Two-Dimensional Compressed Sensing

GUI application demonstrating compressed sensing for image reconstruction.
Recovers full images from sparse pixel samples using L1-regularized optimization in the 2D DCT domain.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import scipy.fftpack as spfft
import scipy.optimize as spopt
import threading


class CompressedSensingApp:
    """
    GUI application for compressed sensing demonstration.
    
    Operations:
        - Upload: Load an image
        - Sample: Randomly sample a percentage of pixels
        - Reconstruct: Recover the full image from sparse samples
    """
    def __init__(self, root):
        """Initialize the application with GUI components."""
        self.root = root
        self.root.title("Two-Dimensional Compressed Sensing")
        
        # Create placeholder image to ensure proper sizing
        placeholder = Image.new('RGB', (320, 400), color='gray')
        self.placeholder_photo = ImageTk.PhotoImage(placeholder)
        
        # Store image data
        self.original_image = None
        self.original_array = None
        self.sample_mask = None
        
        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)
        
        # Create container for first box with upload button
        first_container = tk.Frame(main_frame)
        first_container.pack(side=tk.LEFT, padx=5)
        
        # Upload button above first box
        upload_btn = tk.Button(first_container, text="Upload Image", command=self.upload_image)
        upload_btn.pack(pady=(0, 5))
        
        # First image box
        first_box = tk.Label(first_container, image=self.placeholder_photo, bg='gray', relief='solid', borderwidth=2)
        first_box.pack()
        
        # Create container for second box with sampling controls
        second_container = tk.Frame(main_frame)
        second_container.pack(side=tk.LEFT, padx=5)
        
        # Sampling controls above second box
        sample_controls = tk.Frame(second_container)
        sample_controls.pack(pady=(0, 5))
        
        tk.Label(sample_controls, text="Sample:").pack(side=tk.LEFT, padx=(0, 5))
        self.sample_entry = tk.Entry(sample_controls, width=8)
        self.sample_entry.insert(0, "0.1")
        self.sample_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        sample_btn = tk.Button(sample_controls, text="Sample", command=self.sample_image)
        sample_btn.pack(side=tk.LEFT)
        
        # Second image box
        second_box = tk.Label(second_container, image=self.placeholder_photo, bg='gray', relief='solid', borderwidth=2)
        second_box.pack()
        
        # Create container for third box with reconstruction button
        third_container = tk.Frame(main_frame)
        third_container.pack(side=tk.LEFT, padx=5)
        
        # Reconstruction controls above third box
        reconstruct_controls = tk.Frame(third_container)
        reconstruct_controls.pack(pady=(0, 5))
        
        reconstruct_btn = tk.Button(reconstruct_controls, text="Reconstruct", command=self.reconstruct_image)
        reconstruct_btn.pack(side=tk.LEFT)
        
        self.loading_label = tk.Label(reconstruct_controls, text="", width=2)
        self.loading_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Third image box
        third_box = tk.Label(third_container, image=self.placeholder_photo, bg='gray', relief='solid', borderwidth=2)
        third_box.pack()
        
        # Store references to image display boxes
        self.boxes = [first_box, second_box, third_box]
        
        # Loading animation state
        self.loading_active = False
        self.loading_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.loading_index = 0
    
    def upload_image(self):
        """Open file dialog to load an image (JPG, PNG, BMP, GIF, TIFF)."""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image(file_path, 0)
    
    def sample_image(self):
        """Randomly sample pixels from the image at the specified rate (0.0-1.0)."""
        if self.original_image is None:
            tk.messagebox.showwarning("No Image", "Please upload an image first.")
            return
        
        try:
            sample_rate = float(self.sample_entry.get())
            if sample_rate < 0.0 or sample_rate > 1.0:
                tk.messagebox.showerror("Invalid Value", "Sample value must be between 0.00 and 1.00")
                return
        except ValueError:
            tk.messagebox.showerror("Invalid Value", "Please enter a valid number")
            return
        
        # Convert image to numpy array
        img_array = np.array(self.original_image)
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        # Calculate number of pixels to sample
        num_samples = int(total_pixels * sample_rate)
        
        # Create random sample of indices
        sample_indices = np.random.choice(total_pixels, num_samples, replace=False)
        
        # Store the original image array for reconstruction
        self.original_array = img_array.copy()
        
        # Create a boolean mask
        mask = np.zeros(total_pixels, dtype=bool)
        mask[sample_indices] = True
        mask = mask.reshape(height, width)
        self.sample_mask = mask
        
        # Create visualization image with white background
        sampled_array = np.full_like(img_array, 255)
        if len(img_array.shape) == 3:
            for i in range(img_array.shape[2]):
                sampled_array[:, :, i] = np.where(mask, img_array[:, :, i], 255)
        else:
            sampled_array = np.where(mask, img_array, 255)
        
        # Display sampled image in second box
        sampled_image = Image.fromarray(sampled_array.astype('uint8'))
        img_resized = self.resize_image(sampled_image, 320, 400)
        photo = ImageTk.PhotoImage(img_resized)
        
        self.boxes[1].config(image=photo)
        self.boxes[1].image = photo
    
    def reconstruct_image(self):
        """Reconstruct the full image from sampled pixels using L1-minimization in DCT domain."""
        if self.original_array is None or self.sample_mask is None:
            messagebox.showwarning("No Sampled Image", "Please sample an image first.")
            return
        
        # Start loading animation
        self.loading_active = True
        self.loading_index = 0
        self._update_loading()
        
        # Run reconstruction in separate thread to keep GUI responsive
        thread = threading.Thread(target=self._perform_reconstruction, daemon=True)
        thread.start()
    
    def _perform_reconstruction(self):
        """Perform reconstruction in a separate thread to keep GUI responsive."""
        # Get original image data
        X = self.original_array
        
        # Process each channel separately for color images
        if len(X.shape) == 3:
            ny, nx, nchan = X.shape
            reconstructed = np.zeros(X.shape, dtype='uint8')
            
            for j in range(nchan):
                # Extract channel
                X_channel = X[:, :, j].squeeze()
                
                # Reconstruct this channel
                reconstructed[:, :, j] = self._reconstruct_channel(
                    X_channel, self.sample_mask
                )
        else:
            # Grayscale image
            reconstructed = self._reconstruct_channel(X, self.sample_mask)
        
        # Update GUI in main thread
        self.root.after(0, lambda: self._display_reconstruction(reconstructed))
    
    def _display_reconstruction(self, reconstructed):
        """Display the reconstructed image and stop loading animation."""
        # Display reconstructed image in third box
        reconstructed_image = Image.fromarray(reconstructed.astype('uint8'))
        img_resized = self.resize_image(reconstructed_image, 320, 400)
        photo = ImageTk.PhotoImage(img_resized)
        
        self.boxes[2].config(image=photo)
        self.boxes[2].image = photo
        
        # Stop loading animation
        self.loading_active = False
        self.loading_label.config(text="")
    
    def _dct2(self, x):
        """Compute 2D Discrete Cosine Transform."""
        return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
    
    def _idct2(self, x):
        """Compute 2D Inverse Discrete Cosine Transform."""
        return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
    
    def _reconstruct_channel(self, channel, mask):
        """
        Reconstruct a single channel using L-BFGS-B optimization in DCT domain.
        
        Args:
            channel: 2D array of image channel
            mask: Boolean mask of sampled pixel locations
            
        Returns:
            Reconstructed channel array (values in [0, 255])
        """
        ny, nx = channel.shape
        
        # Get sampled pixel values and indices
        ri = np.where(mask.T.flatten())[0]  # indices of sampled pixels
        b = channel.T.flat[ri].astype(np.float64)
        
        # L1 regularization parameter
        lambda_l1 = 0.1
        
        # Define objective function and gradient for scipy optimizer
        def objective_and_grad(x):
            """Compute objective (L2 loss + L1 regularization) and gradient."""
            # Reshape x to 2D (expand columns-first)
            x2 = x.reshape((nx, ny)).T
            
            # Ax is the inverse 2D DCT of x2
            Ax2 = self._idct2(x2)
            
            # Stack columns and extract samples
            Ax = Ax2.T.flat[ri]
            
            # Calculate residual Ax - b and its 2-norm squared
            Axb = Ax - b
            fx_l2 = np.sum(np.power(Axb, 2))
            fx_l1 = lambda_l1 * np.sum(np.abs(x))
            fx = fx_l2 + fx_l1
            
            # Compute gradient of L2 term
            # Project residual vector onto blank image
            Axb2 = np.zeros((ny, nx))
            Axb2.T.flat[ri] = Axb
            
            # A'(Ax-b) is the 2D DCT of Axb2
            AtAxb2 = 2 * self._dct2(Axb2)
            gx_l2 = AtAxb2.T.flatten()
            
            # Gradient of L1 term (subgradient)
            gx_l1 = lambda_l1 * np.sign(x)
            
            gx = gx_l2 + gx_l1
            
            return fx, gx
        
        # Initial guess (zeros in frequency domain)
        x0 = np.zeros(nx * ny)
        
        # Perform optimization using L-BFGS-B
        result = spopt.minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'disp': False}
        )
        
        # Transform output back to spatial domain
        Xat = result.x.reshape(nx, ny).T
        Xa = self._idct2(Xat)
        
        return np.clip(Xa, 0, 255)
    
    def _update_loading(self):
        """Update the loading spinner animation."""
        if self.loading_active:
            self.loading_label.config(text=self.loading_frames[self.loading_index])
            self.loading_index = (self.loading_index + 1) % len(self.loading_frames)
            self.root.after(100, self._update_loading)
    
    def display_image(self, image_path, box_index):
        """Display an image in the specified box (0=original, 1=sampled, 2=reconstructed)."""
        if box_index < 0 or box_index >= 3:
            return
        
        img = Image.open(image_path)
        img_resized = self.resize_image(img, 320, 400)
        photo = ImageTk.PhotoImage(img_resized)
        
        self.boxes[box_index].config(image=photo)
        self.boxes[box_index].image = photo
    
    def resize_image(self, img, max_width, max_height):
        """Resize image to fit within dimensions while maintaining aspect ratio."""
        width, height = img.size
        aspect_ratio = width / height
        target_ratio = max_width / max_height
        
        if aspect_ratio > target_ratio:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        
        return img.resize((new_width, new_height), Image.LANCZOS)


if __name__ == "__main__":
    root = tk.Tk()
    app = CompressedSensingApp(root)
    root.mainloop()