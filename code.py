import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import random
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load or Train ML Model
try:
    model = joblib.load("image_classifier.pkl")
except:
    model = None

# GUI to Select Image Type
def select_image_type():
    def set_color():
        root.destroy()
        process_color_image()

    def set_bw():
        root.destroy()
        process_bw_image()

    global root
    root = tk.Tk()
    root.title("Image Classification")
    root.state('zoomed')  # Maximize window

    # Add KIIT Logo
    logo_path = "C:/Users/KIIT/Desktop/MINI PROJECT/download.png"
    logo_img = Image.open(logo_path)
    logo_img = logo_img.resize((200, 80), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = Label(root, image=logo_photo)
    logo_label.image = logo_photo
    logo_label.pack(pady=20)

    label = Label(root, text="Welcome!\n\nPlease classify your original image in any of the category which is to be enhanced",
                  font=("Arial", 24))
    label.pack(pady=20)

    btn_color = Button(root, text="Color", command=set_color, bg="lightgreen", font=("Arial", 20), width=20, height=3)
    btn_color.pack(pady=20)

    btn_bw = Button(root, text="Black & White", command=set_bw, bg="lightblue", font=("Arial", 20), width=20, height=3)
    btn_bw.pack(pady=20)

    root.mainloop()

# Processing Color Images
def process_color_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image = cv2.imread(file_path)
    enhanced_image, category = enhance_color_image(image)
    output_path = f"enhanced_{category.lower().replace(' ', '_')}.jpg"
    cv2.imwrite(output_path, enhanced_image)
    display_image(output_path)

# Processing Black & White Images
def process_bw_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not file_path:
        return

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return

    pso = PSO(image)
    best_params = pso.optimize()
    enhanced_image = enhance_bw_image(image, *best_params)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('Enhanced Image')
    plt.show()

# Fitness Function: Measures edge intensity, number of edges, and entropy
def fitness_function(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)

    edge_intensity = np.sum(edges)
    edge_count = np.sum(edges > 50)
    entropy = -np.sum((image / 255.0) * np.log2(image / 255.0 + 1e-10))

    edge_intensity = max(edge_intensity, 1e-5)  # Avoid log(0) issues
    return np.log(np.log(edge_intensity) + 1.00001) * edge_count * entropy  # Ensure positive value inside log

# B&W Image Enhancement using PSO
class PSO:
    def __init__(self, image, num_particles=30, max_iter=200):
        self.image = image
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.particles = [np.random.uniform(0.5, 2.0, 4) for _ in range(num_particles)]
        self.velocities = [np.random.uniform(-0.1, 0.1, 4) for _ in range(num_particles)]
        self.p_best = self.particles[:]
        self.g_best = max(self.particles, key=lambda p: fitness_function(enhance_bw_image(image, *p)))
        self.w = 0.7
        self.c1, self.c2 = 1.3, 1.3

    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                enhanced_image = enhance_bw_image(self.image, *self.particles[i])
                fitness = fitness_function(enhanced_image)

                if fitness > fitness_function(enhance_bw_image(self.image, *self.p_best[i])):
                    self.p_best[i] = self.particles[i]
                if fitness > fitness_function(enhance_bw_image(self.image, *self.g_best)):
                    self.g_best = self.particles[i]

                r1, r2 = random.random(), random.random()
                self.velocities[i] = (self.w * np.array(self.velocities[i]) +
                                      self.c1 * r1 * (np.array(self.p_best[i]) - np.array(self.particles[i])) +
                                      self.c2 * r2 * (np.array(self.g_best) - np.array(self.particles[i])))
                self.particles[i] = np.clip(np.array(self.particles[i]) + np.array(self.velocities[i]), 0.5, 2.0)
        return self.g_best

# B&W Enhancement Function
def enhance_bw_image(image, a, b, c, k):
    mean_local = cv2.blur(image, (3, 3))
    std_local = cv2.Laplacian(image, cv2.CV_64F).var()
    enhanced = ((k * mean_local / (std_local + 1e-10) + b) * (image - c * mean_local) + mean_local * a)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

# Color Image Enhancement Function
def enhance_color_image(image):
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return enhanced, "Color"

# Display Image
def display_image(image_path):
    root_display = tk.Tk()
    root_display.title("Enhanced Image")
    img = Image.open(image_path)
    img = img.resize((400, 300), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    label = Label(root_display, image=img)
    label.image = img
    label.pack()
    root_display.mainloop()

# Run Main GUI
if __name__ == "__main__":
    select_image_type()
    