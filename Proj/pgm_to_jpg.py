from PIL import Image

img = Image.open("Mona_Lisa_out.pgm")
img.save("Mona_Lisa_out.jpg", "JPEG")

print("Done")