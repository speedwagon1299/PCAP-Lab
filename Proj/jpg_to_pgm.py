from PIL import Image

img = Image.open("Mona_Lisa.jpg")
img = img.convert("L")
img.save("Mona_Lisa.pgm")
