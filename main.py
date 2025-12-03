"""
Two-Dimensional Compressed Sensing Application

This application demonstrates compressed sensing for image reconstruction using a 2D DCT basis.
Users can upload an image, randomly sample a subset of pixels, and reconstruct the full image
from the sparse samples using L1 minimization.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import scipy.fft as spfft
import threading


class CompressedSensingApp:
    """
    A GUI application for demonstrating compressed sensing on images.
    
    The application provides three main operations:
    1. Upload: Load an image from disk
    2. Sample: Randomly sample a percentage of pixels from the image
    3. Reconstruct: Recover the full image from sampled pixels using compressed sensing
    
    Attributes:
        root (tk.Tk): The main Tkinter window
        original_image (PIL.Image): The uploaded original image
        sampled_array (np.ndarray): Array representation of the sampled image
        sample_mask (np.ndarray): Boolean mask indicating sampled pixel locations
        sample_indices (np.ndarray): Indices of sampled pixels
        boxes (list): List of label widgets displaying the three image stages
        placeholder_photo (ImageTk.PhotoImage): Placeholder image for empty boxes
        sample_entry (tk.Entry): Entry widget for sample rate input
    """
    def __init__(self, root):
        """
        Initialize the compressed sensing application.
        
        Args:
            root (tk.Tk): The main Tkinter window
        """
        self.root = root
        self.root.title("Two-Dimensional Compressed Sensing")
        
        # Create placeholder image to ensure proper sizing
        placeholder = Image.new('RGB', (320, 400), color='gray')
        self.placeholder_photo = ImageTk.PhotoImage(placeholder)
        
        # Store image data
        self.original_image = None
        self.sampled_array = None
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
        """
        Open a file dialog to select and load an image.
        
        Displays the selected image in the first box and stores it for subsequent operations.
        Supports common image formats including JPG, PNG, BMP, GIF, and TIFF.
        """
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image(file_path, 0)
    
    def sample_image(self):
        """
        Randomly sample pixels from the original image based on the specified sample rate.
        
        Creates a sparse representation by selecting a random subset of pixels and displaying
        them on a white background. The sample rate is read from the entry widget and must be
        between 0.0 and 1.0. Sampled pixels and their locations are stored for reconstruction.
        
        Raises:
            ValueError: If the sample rate is not a valid number
            Warning: If no image has been uploaded or sample rate is out of bounds
        """
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
        
        # Create a mask of sampled pixel locations
        mask = np.zeros(total_pixels, dtype=bool)
        sample_indices = np.random.choice(total_pixels, num_samples, replace=False)
        mask[sample_indices] = True
        mask = mask.reshape(height, width)
        
        # Store sampling information for reconstruction
        self.sample_mask = mask
        
        # Create sampled image with white background
        sampled_array = np.full_like(img_array, 255)
        if len(img_array.shape) == 3:
            for i in range(img_array.shape[2]):
                sampled_array[:, :, i] = np.where(mask, img_array[:, :, i], 255)
        else:
            sampled_array = np.where(mask, img_array, 255)
        
        self.sampled_array = sampled_array
        
        # Display sampled image in second box
        sampled_image = Image.fromarray(sampled_array.astype('uint8'))
        img_resized = self.resize_image(sampled_image, 320, 400)
        photo = ImageTk.PhotoImage(img_resized)
        
        self.boxes[1].config(image=photo)
        self.boxes[1].image = photo
    
    def reconstruct_image(self):
        """
        Reconstruct the full image from sampled pixels using compressed sensing.
        
        Uses a 2D Discrete Cosine Transform (DCT) basis for sparse representation and
        L1 minimization to recover the complete image from the subset of sampled pixels.
        The reconstruction is performed independently for each color channel in RGB images.
        
        This process may take several seconds depending on image size and sampling rate.
        
        Raises:
            Warning: If no sampled image exists
        """
        if self.sampled_array is None or self.sample_mask is None:
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
        """
        Perform the actual reconstruction in a separate thread.
        This keeps the GUI responsive and allows the loading animation to play.
        """
        # Process each channel separately for color images
        if len(self.sampled_array.shape) == 3:
            reconstructed = np.zeros_like(self.sampled_array)
            for c in range(self.sampled_array.shape[2]):
                reconstructed[:, :, c] = self._reconstruct_channel(
                    self.sampled_array[:, :, c], self.sample_mask
                )
        else:
            reconstructed = self._reconstruct_channel(self.sampled_array, self.sample_mask)
        
        # Update GUI in main thread
        self.root.after(0, lambda: self._display_reconstruction(reconstructed))
    
    def _display_reconstruction(self, reconstructed):
        """
        Display the reconstructed image and stop the loading animation.
        Must be called from the main GUI thread.
        
        Args:
            reconstructed (np.ndarray): The reconstructed image array
        """
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
        """2D Discrete Cosine Transform."""
        return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
    
    def _idct2(self, x):
        """2D Inverse Discrete Cosine Transform."""
        return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
    
    def _reconstruct_channel(self, channel, mask):
        """
        Reconstruct a single color channel using compressed sensing.
        
        Performs L1 optimization in the 2D DCT domain to recover the full
        channel from sampled measurements using the standard compressed sensing approach.
        
        Args:
            channel (np.ndarray): 2D array representing a single image channel (sampled image with white background)
            mask (np.ndarray): Boolean array indicating sampled pixel locations
            
        Returns:
            np.ndarray: Reconstructed channel with values clipped to [0, 255]
        """
        import cvxpy as cvx
        
        ny, nx = channel.shape
        
        # Get sampled pixel values and indices
        ri = np.where(mask.T.flatten())[0]  # indices of sampled pixels
        b = channel.T.flat[ri].astype(np.float64)
        b = np.expand_dims(b, axis=1)
        
        # Create DCT measurement matrix using Kronecker product
        A = np.kron(
            spfft.idct(np.identity(nx), norm='ortho', axis=0),
            spfft.idct(np.identity(ny), norm='ortho', axis=0)
        )
        A = A[ri, :]  # select only sampled rows
        
        # Solve L1 minimization problem
        vx = cvx.Variable(nx * ny)
        objective = cvx.Minimize(cvx.norm(vx, 1))
        constraints = [A @ vx == b.flatten()]
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(verbose=False)
        
        # Reconstruct signal
        Xat = np.array(vx.value).squeeze()
        Xat = Xat.reshape(nx, ny).T
        Xa = self._idct2(Xat)
        
        return np.clip(Xa, 0, 255)
    
    def _update_loading(self):
        """Update the loading spinner animation."""
        if self.loading_active:
            self.loading_label.config(text=self.loading_frames[self.loading_index])
            self.loading_index = (self.loading_index + 1) % len(self.loading_frames)
            self.root.after(100, self._update_loading)
    
    def display_image(self, image_path, box_index):
        """
        Display an image in one of the three display boxes.
        
        Args:
            image_path (str): Path to the image file
            box_index (int): Index of the box (0=original, 1=sampled, 2=reconstructed)
        """
        if box_index < 0 or box_index >= 3:
            return
        
        img = Image.open(image_path)
        img_resized = self.resize_image(img, 320, 400)
        photo = ImageTk.PhotoImage(img_resized)
        
        self.boxes[box_index].config(image=photo)
        self.boxes[box_index].image = photo
    
    def resize_image(self, img, max_width, max_height):
        """
        Resize an image to fill specified dimensions while maintaining aspect ratio.
        
        Images are scaled to fill the box as much as possible without distortion.
        
        Args:
            img (PIL.Image): Image to resize
            max_width (int): Target width in pixels
            max_height (int): Target height in pixels
            
        Returns:
            PIL.Image: Resized image
        """
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