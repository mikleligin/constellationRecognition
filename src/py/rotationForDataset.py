from PIL import Image
 
im = Image.open('1.jpg')
for i in range(360):
    im_rotate = im.rotate(i)
    im_rotate.save(f'swan{i}.jpg', quality=100)
im.close()
