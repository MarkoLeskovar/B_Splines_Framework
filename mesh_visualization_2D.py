import numpy as np
import matplotlib.pyplot as plt
import custom_functions as cf



# INFO
# This program is a example of mesh visualization and voxel indexing in 2D
# It uses matplotlib visualization module
if __name__ == "__main__":

    # Initialize image pixel dimensions
    im_dir_1 = 6 # x-axis (width)
    im_dir_2 = 4 # y-axis (height)

    # Voxel size
    im_size_1 = 1
    im_size_2 = 1

    # Image width and height
    im_width = im_dir_1 * im_size_1
    im_height = im_dir_2 * im_size_2

    # Create a range of grid coordinates for pixel vertices and centers
    x_range_pixel_vertices = np.linspace(-0.5 * im_size_1, (im_dir_1 - 0.5) * im_size_1, im_dir_1 + 1)
    y_range_pixel_vertices = np.linspace(-0.5 * im_size_2, (im_dir_2 - 0.5) * im_size_2, im_dir_2 + 1)
    x_range_pixel_centers = np.linspace(0, (im_dir_1 - 1) * im_size_1, im_dir_1)
    y_range_pixel_centers = np.linspace(0, (im_dir_2 - 1) * im_size_2, im_dir_2)

    # Create grid coordinates for pixel vertices and centers
    x_grid_pixel_vertices, y_grid_pixel_vertices = np.meshgrid(x_range_pixel_vertices, y_range_pixel_vertices, indexing='ij')
    x_grid_pixel_centers, y_grid_pixel_centers = np.meshgrid(x_range_pixel_centers, y_range_pixel_centers, indexing='ij')

    # Create an uniform mesh
    mesh = cf.uniformRectangularMesh2D(x_grid_pixel_vertices, y_grid_pixel_vertices)

    # element_index = mesh.getElementIndex(3, 2)
    # node_index = mesh.elements[element_index]

    # Create a dummy binary mask
    binary_mask = np.ones((6,4))
    binary_mask[1, 1] = 0
    binary_mask[2, 1] = 0
    binary_mask[3, 1] = 0
    binary_mask[1, 2] = 0
    binary_mask[2, 2] = 0



    # Initialize the figure
    fig, ax = plt.subplots()
    # ax.set_xlim(0, im_width)
    # ax.set_ylim(0, im_height)
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_aspect('equal')

    # Apply the mask to data
    fat_pad = np.ma.masked_array(mesh.element_areas, binary_mask)

    plt.pcolormesh(x_grid_pixel_vertices, y_grid_pixel_vertices, fat_pad, shading='flat', edgecolors='black', alpha=0.5)
    plt.colorbar()
    plt.show()





    # Draw the entire mesh
    # print('Plotting the entire mesh...')
    # t0 = time.time()
    # for n in range(mesh.num_edges):
    #
    #     # Get individual node ids from element list
    #     edge = mesh.edges[n, :]
    #     start_node = mesh.nodes[edge[0], :]
    #     end_node   = mesh.nodes[edge[1], :]
    #
    #     # Plot each edge
    #     plt.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], '-b')
    #
    # print('...done! t =', "{:.1f}".format(time.time()-t0), 's\n')
    # plt.show()
