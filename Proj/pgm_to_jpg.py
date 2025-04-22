from PIL import Image

img = Image.open("Mona_Lisa_sobel_cu.pgm")
img.save("Mona_Lisa_sobel_cu.jpg", "JPEG")
