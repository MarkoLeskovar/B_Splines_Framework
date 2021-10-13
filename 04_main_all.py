import os
import time
import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import custom_functions as cf


if __name__ == "__main__":

    # Elastix and Tranformix system paths
    path_to_elastix = '/home/mleskova/install_dir/elastix-5.0.1-linux/install/bin/elastix'
    path_to_transformix = '/home/mleskova/install_dir/elastix-5.0.1-linux/install/bin/transformix'

    # Path to data relative to current working directory
    path_to_input = 'dummy_data_2D/square'
    path_to_output = 'output_dummy_data_2D/square'

    # Load path to images, output and parameter file
    source_image = '03_source.png'
    target_image = '03_target.png'

    # Load the parameter file
    # WARNING! both the center of rotation and image origin should be at [0,0] as they are not included in the equations
    parameter_file = 'parameters_BSpline_4.txt'

    # Get the path to working directory, input and output
    path_to_working_directory = os.getcwd()
    path_to_input = os.path.join(path_to_working_directory, path_to_input)
    path_to_output = os.path.join(path_to_working_directory, path_to_output)
    path_to_bone_output = os.path.join(path_to_output, 'elastix_reg_all')
    transform_parameters_file = os.path.join(path_to_bone_output, 'TransformParameters.0.txt')

    # FILE OPTIONS
    delete_files = False


    # Elastix transform output  : fixed     -> moving
    # What we want              : source    -> target
    # Forward tracking          : 01        -> 02
    fixed_image = source_image
    moving_image = target_image

    # Check if output folder already exists
    if not os.path.exists(path_to_bone_output):

        # Make a directory
        os.makedirs(path_to_bone_output)

        # Combine the elastix input command
        elastix_command = path_to_elastix + \
                          ' -f ' + os.path.join(path_to_input, fixed_image) + \
                          ' -m ' + os.path.join(path_to_input, moving_image) + \
                          ' -p ' + os.path.join(path_to_input, parameter_file) + \
                          ' -out ' + path_to_bone_output + ' > log.txt'

        # Run  elastix in command line
        print("Running elastix registration... ")
        t0 = time.time()

        # Call elastix
        # print(elastix_command)
        os.system(elastix_command)

        print("...done! t =", "{:.1f}".format(time.time() - t0), 's\n')

        # Read in the mhd image and save it as png
        result_im = cf.mhd_image(os.path.join(path_to_bone_output, 'result.0.mhd'))
        result_im.convertToPNG(os.path.join(path_to_output, 'result.0.png'))

        # DELETE UNNECESSARY FILES
        if delete_files:
            os.remove(os.path.join(path_to_bone_output, 'result.0.mhd'))
            os.remove(os.path.join(path_to_bone_output, 'result.0.raw'))

    else:
        print("Output folder already exists! -> ", path_to_bone_output)



    # Load images
    source_im = imageio.imread(os.path.join(path_to_input, source_image)).transpose()
    target_im = imageio.imread(os.path.join(path_to_input, target_image)).transpose()
    result_im = imageio.imread(os.path.join(path_to_output, 'result.0.png')).transpose()

    # Load soft tissue masks for time frame 00 and cut of last row and column
    # mask_example = imageio.imread(os.path.join(path_to_input, 'mask_example.png')).transpose()
    # Cut-off last row and column
    # mask_example = mask_example[:-1, :-1]
    # Change the mask to 0-1 integer values
    # mask_example = (mask_example / mask_example.max()).astype('int')







    # Information about the image size
    im_dir_1 = source_im.shape[0] # x-direction (width)
    im_dir_2 = source_im.shape[1] # y-direction (height)
    im_length = im_dir_1 * im_dir_2

    # Voxel size
    im_size_1 = 1
    im_size_2 = 1

    # Create a grid of pixel coordinates
    x_range = np.linspace(0., (im_dir_1 - 1) * im_size_1, im_dir_1)
    y_range = np.linspace(0., (im_dir_2 - 1) * im_size_2, im_dir_2)

    # Create a 2D meshgrid of coordinates
    source_coordinates_x, source_coordinates_y = np.meshgrid(x_range, y_range, indexing='ij')

    # Flatten the 2D grid to 1D array
    x_points = source_coordinates_x.flatten()
    y_points = source_coordinates_y.flatten()

    # Path to text file
    path_to_input_file = os.path.join(path_to_output, 'inputpoints.txt')

    # Write a text file
    file = open(path_to_input_file, 'w')       # Overwrite content
    file.write('point\n')                   # Tell elastix that we are dealing with physical points
    file.write(str(im_length) + '\n')       # Number of points
    for x, y in zip(x_points, y_points):
        file.write("{:.1f}".format(x) + '\t' + "{:.1f}".format(y) + '\n')
    file.close()

    # Combine the transformix input command
    transformix_command = path_to_transformix + \
                          ' -def ' + path_to_input_file + \
                          ' -out ' + path_to_output + \
                          ' -tp ' + transform_parameters_file + ' > log.txt'

    # Call transformix
    # print(transformix_command)
    os.system(transformix_command)

    # Initialize output
    target_coordinates_x = np.zeros(im_length, dtype="float32")
    target_coordinates_y = np.zeros(im_length, dtype="float32")

    # Name of the output file
    path_to_output_file = os.path.join(path_to_output, 'outputpoints.txt')

    # Read in the data
    file = open(path_to_output_file, 'r')
    lines = file.readlines()

    # Find the transformed coordinates
    k = 0
    for line in lines:

        # Get the data from each line
        transformed_points = line.split('\t')
        transformed_points = transformed_points[5]
        transformed_points = transformed_points.split(' ')

        # Assign the x and y coordinate
        target_coordinates_x[k] = float(transformed_points[4])
        target_coordinates_y[k] = float(transformed_points[5])

        # Update the counter
        k += 1




    # Reshape the 1D array to get the original 2D matrix format
    target_coordinates_x = np.reshape(target_coordinates_x, (im_dir_1, im_dir_2))
    target_coordinates_y = np.reshape(target_coordinates_y, (im_dir_1, im_dir_2))



    # Get the relative displacement field
    displacement_field_x = target_coordinates_x - source_coordinates_x
    displacement_field_y = target_coordinates_y - source_coordinates_y

    # Compute the gradient of the displacement field
    gx_x, gx_y = np.gradient(displacement_field_x)
    gy_x, gy_y = np.gradient(displacement_field_y)

    # Add 1 because displacement field is relative: y = x + w(x), so dy/dx = 1 + dw(x)/dx
    gx_x += 1.0
    gy_y += 1.0

    # Apply Gaussian filter to the gradient field
    gaussian_filter(gx_x, sigma=2, output=gx_x)
    gaussian_filter(gx_y, sigma=2, output=gx_y)
    gaussian_filter(gy_x, sigma=2, output=gy_x)
    gaussian_filter(gy_y, sigma=2, output=gy_y)

    # Compute the determinant of the Jacobian for each entry
    jacobian_determinant = ((gx_x * gy_y) - (gy_x * gx_y))

    # Compute element areas for each time frame
    element_areas = cf.computeElementAreas2D(target_coordinates_x, target_coordinates_y)

    # Apply Gaussian filter to the areas field
    gaussian_filter(element_areas, sigma=2, output=element_areas)



    # Get only the relative change in volume
    # jacobian_determinant -= 1.0
    element_areas -= 1.0

    # Get the maximum and minimum of jacobian
    jacobian_min = jacobian_determinant.min()
    jacobian_max = jacobian_determinant.max()

    # Get the maximum and minimum of element areas
    element_areas_min = element_areas.min()
    element_areas_max = element_areas.max()


    # DELETE UNNECESSARY FILES
    if delete_files:
        os.remove(path_to_input_file)
        os.remove(path_to_output_file)



    # Downsample the images for plotting
    downsampling = 10
    u_x = displacement_field_x[::downsampling, ::downsampling]
    u_y = displacement_field_y[::downsampling, ::downsampling]
    x_grid_target = target_coordinates_x[::downsampling, ::downsampling]
    y_grid_target = target_coordinates_y[::downsampling, ::downsampling]
    x_grid_source = source_coordinates_x[::downsampling, ::downsampling]
    y_grid_source = source_coordinates_y[::downsampling, ::downsampling]



    # PLOTTING OPTIONS
    plot_deformation_mesh = True
    plot_deformation_vectors = True
    plot_volume_change = True
    plot_jacobian = True



    # PLOTTING THE DEFORMATION MESH
    if plot_deformation_mesh == True:

        line_width = 0.5
        marker_size = 0

        print('\n')
        print('-' * 80)
        print('Plotting Deformation Mesh...')

        # Initialize the figure
        fig, ax = plt.subplots()
        # ax.set_xlim(0, target_coordinates_x.max())
        # ax.set_ylim(0, target_coordinates_y.max())
        plt.gca().invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Z-coordinate')
        ax.set_aspect('equal')

        # Show both images
        ax.imshow(source_im.transpose(), cmap=plt.cm.Blues, interpolation='none', alpha=0.5)
        ax.imshow(target_im.transpose(), cmap=plt.cm.Reds, interpolation='none', alpha=0.5)

        # Plot the undeformed grid lines
        for x, y in zip(x_grid_source, y_grid_source):
            plt.plot(x, y, '-o', color='blue', linewidth=line_width, markersize=marker_size)
        for x, y in zip(np.transpose(x_grid_source), np.transpose(y_grid_source)):
            plt.plot(x, y, '-o', color='blue', linewidth=line_width, markersize=marker_size)

        # Plot the deformed grid lines
        for x, y in zip(x_grid_target, y_grid_target):
            plt.plot(x, y, '-o', color='red', linewidth=line_width, markersize=marker_size)
        for x, y in zip(np.transpose(x_grid_target), np.transpose(y_grid_target)):
            plt.plot(x, y, '-o', color='red', linewidth=line_width, markersize=marker_size)



        # DEBUG - plot deformation vectors
        # ax.quiver(x_grid_source, y_grid_source, u_x, u_y, angles='xy', scale_units='xy', scale=1, color='green', width=0.002)

        # fig.show()
        fig.savefig(os.path.join(path_to_output, 'deformation_mesh.png'), dpi=300)
        plt.close(fig)



    # PLOTING THE DEFORMATION VECTORS
    if plot_deformation_vectors == True:

        line_width = 0.5
        marker_size = 0

        print('\n')
        print('-' * 80)
        print('Plotting Displacement Field...')

        # Initialize the figure
        fig, ax = plt.subplots()
        # ax.set_xlim(0, target_coordinates_x.max())
        # ax.set_ylim(0, target_coordinates_y.max())
        plt.gca().invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Z-coordinate')
        ax.set_aspect('equal')

        # Plot the vector field
        ax.quiver(x_grid_source, y_grid_source, u_x, u_y, angles='xy', scale_units='xy', scale=1, color='green', width=0.002)

        # fig.show()
        fig.savefig(os.path.join(path_to_output, 'displacement_field.png'), dpi=300)
        plt.close(fig)



    # PLOTTING OF THE VOLUME CHANGE
    if plot_volume_change == True:

        line_width = 0.5
        marker_size = 0

        print('\n')
        print('-' * 80)
        print('Plotting Relative Volume Change...')

        # Initialize the figure
        fig, ax = plt.subplots()
        # ax.set_xlim(0, target_coordinates_x.max())
        # ax.set_ylim(0, target_coordinates_y.max())
        plt.gca().invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Z-coordinate')
        ax.set_aspect('equal')


        # Define an area to plot
        # area_to_plot = mask_example

        # Create a binary mask
        # binary_mask = np.ma.masked_where(area_to_plot == 0, element_areas)

        # plt.pcolormesh(target_coordinates_x, target_coordinates_y, binary_mask, shading='flat', cmap=plt.cm.jet, edgecolors='none')
        plt.pcolormesh(target_coordinates_x, target_coordinates_y, element_areas, shading='flat', cmap=plt.cm.jet, edgecolors='none')
        plt.colorbar(label="<-compression | relative volume deviation | extension->")
        plt.clim(-element_areas_max, element_areas_max)

        # fig.show()
        fig.savefig(os.path.join(path_to_output, 'volume_deviation.png'), dpi=300)
        plt.close(fig)



    # PLOTING THE DETERMINANT OF JACOBIAN
    if plot_jacobian == True:

        line_width = 0.5
        marker_size = 0

        print('\n')
        print('-' * 80)
        print('Plotting Jacobian...')

        # Initialize the figure
        fig, ax = plt.subplots()
        # ax.set_xlim(0, im_dir_1 - 1)
        # ax.set_ylim(0, im_dir_2 - 1)
        plt.gca().invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Z-coordinate')
        ax.set_aspect('equal')

        img1 = plt.imshow(jacobian_determinant.transpose(), cmap=plt.cm.jet, interpolation='none')  # seismic,PiYG,RdYlGn,hsv
        plt.colorbar()
        # plt.clim(-jacobian_max, jacobian_max)

        # fig.show()
        fig.savefig(os.path.join(path_to_output, 'jacobian_map.png'), dpi=300)
        plt.close(fig)