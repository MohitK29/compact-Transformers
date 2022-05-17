import numpy as np
import cv2
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt

instances = []

# Load in the images
for filepath in os.listdir('vitgan1/samples'):
    instances.append(cv2.imread('vitgan1/samples/{0}'.format(filepath)))

images = np.array(instances)



fig = plt.figure(figsize=(4, 4))
plt.axis("off")
ims = [[plt.imshow(i, animated=True)] for i in instances[::5]]
ani = animation.ArtistAnimation(fig, ims, interval=250, repeat_delay=250, blit=True)
f = './animation.gif'
writergif = animation.PillowWriter(fps=30) 
ani.save(f, writer=writergif)
