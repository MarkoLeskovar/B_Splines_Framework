import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# INFO
# This program crates a .png image that contains a regular rectangular grid
if __name__ == "__main__":

    # Path to data relative to current working directory
    path_to_output = 'output_dummy_data_2D/square'

    # Get the path to working directory, input and output
    path_to_working_directory = os.getcwd()
    path_to_output = os.path.join(path_to_working_directory, path_to_output)

    # Load path to images, output and parameter file
    output_image = '01_grid.png'

    # Information about the image size
    im_dir_1 = 128 # x-direction (width)
    im_dir_2 = 128 # y-direction (height)

    # Define grid size
    grid_size = 10

    # Create an image
    grid_image = np.zeros((im_dir_1,im_dir_2))
    grid_image[:, ::grid_size] = 255
    grid_image[::grid_size, :] = 255
    grid_image = grid_image.astype(np.uint8)

    # Convert and save the image
    # TODO: convert to monochrome. Is it necessary ??
    image = Image.fromarray(grid_image.transpose())
    image = image.convert("L")
    image.save(os.path.join(path_to_output, output_image))

    # DEBUG - show the image
    # plt.imshow(grid_image.transpose(), cmap ='gray', interpolation='none')
    # plt.show()










