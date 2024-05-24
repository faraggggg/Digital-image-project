import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, Canvas, Button, Label, ttk
from PIL import Image, ImageTk

BG_COLOR = "#f0f0f0"
BUTTON_COLOR = "#4CAF50"
LABEL_COLOR = "#333333"


def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        return image
    return None


class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Histogram Analyzer")
        self.root.config(background='black')

        # Create a frame for better organization
        self.frame = ttk.Frame(root, padding="20")
        self.frame.pack()

        # Create Canvas for displaying the image
        self.canvas = Canvas(self.frame, width=800, height=600, background="black")
        self.canvas.grid(row=0, column=0, columnspan=4, pady=10)

        # Load Image Button
        self.load_button = Button(self.frame, text="Load Image", command=self.load_image, bg='darkblue', fg="white")
        self.load_button.grid(row=1, column=0, padx=5, pady=5)

        # Grayscale Button
        self.gray_button = Button(self.frame, text="Grayscale", command=self.convert_to_grayscale, bg=BUTTON_COLOR, fg="white")
        self.gray_button.grid(row=1, column=1, padx=5, pady=5)

        # Enhance Image Button
        self.enhance_button = Button(self.frame, text="Enhance Image", command=self.enhance_image, bg=BUTTON_COLOR, fg="white")
        self.enhance_button.grid(row=1, column=2, padx=5, pady=5)

        # Apply CLAHE Button
        self.clahe_button = Button(self.frame, text="Apply CLAHE", command=self.apply_clahe, bg=BUTTON_COLOR, fg="white")
        self.clahe_button.grid(row=1, column=3, padx=5, pady=5)

        # Apply Gaussian Blur Button
        self.blur_button = Button(self.frame, text="Apply Gaussian Blur", command=self.apply_gaussian_blur, bg=BUTTON_COLOR, fg="white")
        self.blur_button.grid(row=1, column=4, padx=5, pady=5)

        # Show Histogram Button
        self.hist_button = Button(self.frame, text="Show Histogram", command=self.show_histogram, bg='darkblue', fg="white")
        self.hist_button.grid(row=2, column=0, columnspan=4, pady=5)

        # Original Image Button
        self.original_button = Button(self.frame, text="Original Image", command=self.load_original_image, bg='darkblue', fg="white")
        self.original_button.grid(row=3, column=0, columnspan=4, pady=5)

        # Labels for statistics
        self.mean_label = Label(self.frame, text="Mean: N/A", bg='grey', fg='black')
        self.mean_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        self.median_label = Label(self.frame, text="Median: N/A", bg='grey', fg='black')
        self.median_label.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        self.std_dev_label = Label(self.frame, text="Standard Deviation: N/A", bg='grey', fg='black')
        self.std_dev_label.grid(row=4, column=2, padx=5, pady=5, sticky="w")

        self.image = None
        self.original_image = None

    def load_image(self):
        self.image = load_image()
        if self.image is not None:
            self.original_image = self.image.copy()
            self.display_image()
            self.update_statistics(self.image)

    def display_image(self):
        self.canvas.delete("all")
        display_image(self.canvas, self.image, 0, 0)

    def convert_to_grayscale(self):
        if self.image is not None:
            print("Original Image Shape:", self.image.shape)

            if len(self.image.shape) == 3 and self.image.shape[2] == 3:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                print("Converted to Grayscale:", self.image.shape)
                self.display_image()
                self.update_statistics(self.image)
            else:
                print("The image is already in grayscale or does not have the expected number of color channels.")

    def show_histogram(self):
        if self.image is not None:
            plot_histogram(self.image)

    def enhance_image(self):
        if self.image is not None:
            enhanced_image = equalize_histogram(self.image)
            self.image = enhanced_image
            self.display_image()
            self.update_statistics(self.image)

    def apply_clahe(self):
        if self.image is not None:
            print("Original Image Type:", self.image.dtype)
            print("Original Image Shape:", self.image.shape)

            if len(self.image.shape) == 3:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                print("Converted to Grayscale:", gray_image.dtype)
                print("Grayscale Image Shape:", gray_image.shape)
            else:
                gray_image = self.image

            if gray_image.dtype in [np.uint8, np.uint16]:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                self.image = clahe.apply(gray_image)
                print("CLAHE Applied Successfully")
                self.display_image()
                self.update_statistics(self.image)
            else:
                print("Error: Input image type is not supported by CLAHE.")

    def apply_gaussian_blur(self):
        if self.image is not None:
            self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
            self.display_image()
            self.update_statistics(self.image)

    def load_original_image(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.display_image()
            self.update_statistics(self.image)

    def update_statistics(self, image):
        mean, median, std_dev = calculate_statistics(image)
        if isinstance(mean, list):
            mean = np.mean(mean)
            median = np.median(median)
            std_dev = np.mean(std_dev)
        self.mean_label.config(text=f"Mean: {mean:.2f}")
        self.median_label.config(text=f"Median: {median:.2f}")
        self.std_dev_label.config(text=f"Standard Deviation: {std_dev:.2f}")


def display_image(canvas, image, x, y):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    canvas.create_image(x, y, anchor="nw", image=image)
    canvas.image = image


def plot_histogram(image):
    plt.figure(figsize=(10, 5))
    if len(image.shape) == 2:
        plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray')
    else:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()


def calculate_statistics(image):
    if len(image.shape) == 2:
        mean = np.mean(image)
        median = np.median(image)
        std_dev = np.std(image)
    else:
        mean = [np.mean(image[:, :, i]) for i in range(3)]
        median = [np.median(image[:, :, i]) for i in range(3)]
        std_dev = [np.std(image[:, :, i]) for i in range(3)]
    return mean, median, std_dev


def equalize_histogram(image):
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    else:
        channels = cv2.split(image)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq_channels)


if __name__ == "__main__":
    root = Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()
