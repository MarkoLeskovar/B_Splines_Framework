import os
import time
import imageio
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import custom_functions as cf


if __name__ == "__main__":

    # Elastix and Tranformix system paths
    path_to_elastix = '/home/mleskova/install_dir/elastix-5.0.1-linux/install/bin/elastix'
    path_to_transformix = '/home/mleskova/install_dir/elastix-5.0.1-linux/install/bin/transformix'

    # PREAMBLE
    path_to_working_directory = os.getcwd()
    path_to_mri_images = 'dynSeq_data_2D/01_dynSeq'
    path_to_output = 'output_dynSeq_data'

    # Assemble the paths
    path_to_mri_images = os.path.join(path_to_working_directory, path_to_mri_images)
    path_to_output = os.path.join(path_to_working_directory, path_to_output)

    # Load the parameter file
    parameter_file = '/home/mleskova/Workspaces/pycharm/B_Splines_Framework/dynSeq_data_2D/parameters_Bspline.txt'


    # LOAD IMAGE DATA
    # Manually define image dimensions
    im_dir_1 = 170  # width  -> x-axis
    im_dir_2 = 113  # height -> y-axis
    im_length = im_dir_1 * im_dir_2

    # Forward tracking
    time_frames = ['00', '05', '10', '15']
    n_time_frames = [0, 5, 10, 15]
    n_frames = 4



    # FILE OPTIONS
    delete_files = False



    # Create a grid of pixel coordinates
    x_range = np.arange(0, im_dir_1, 1)
    y_range = np.arange(0, im_dir_2, 1)
    source_coordinates_x, source_coordinates_y = np.meshgrid(x_range, y_range, indexing='ij')



    # INITIALIZE VARIABLES
    # Initialize a matrix of physical coordinates for all time frames
    target_coordinates_x = np.zeros((im_dir_1, im_dir_2, n_frames), dtype="float32")
    target_coordinates_y = np.zeros((im_dir_1, im_dir_2, n_frames), dtype="float32")
    target_coordinates_x[:, :, 0] = source_coordinates_x
    target_coordinates_y[:, :, 0] = source_coordinates_y

    # Initialize the relative displacement field
    displacement_field_x = np.zeros((im_dir_1, im_dir_2, n_frames), dtype="float32")
    displacement_field_y = np.zeros((im_dir_1, im_dir_2, n_frames), dtype="float32")
    displacement_field_x[:, :, 0] = 0.0
    displacement_field_y[:, :, 0] = 0.0

    # Initialize the acobian determinant
    jacobian_determinant = np.zeros((im_dir_1, im_dir_2, n_frames), dtype="float32")
    jacobian_determinant[:, :, 0] = 1.0

    # Initialize element areas
    quad_element_areas = np.zeros((im_dir_1 - 1, im_dir_2 - 1, n_frames), dtype="float32")
    quad_element_areas[:, :, 0] = 1.0


    # Check if output folder already exists
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    ####################################################################################################################
    compute_bspline_transform = True

    if compute_bspline_transform == True:
        print("Running elastix registration... ")
        t0 = time.time()

        # Loop over all time frames
        for t in range(n_frames-1):

            # Assign images
            fixed_image = 'dynSeq_' + time_frames[t] + '.png'
            moving_image = 'dynSeq_' + time_frames[t + 1] + '.png'

            # Assign bone output
            path_to_elastix_output = os.path.join(path_to_output, 'elastix_result_' + time_frames[t] + '-' + time_frames[t + 1])
            if not os.path.exists(path_to_elastix_output):
                os.makedirs(path_to_elastix_output)


            # Combine the elastix input command
            elastix_command = path_to_elastix + \
                              ' -f ' + os.path.join(path_to_mri_images, fixed_image) + \
                              ' -m ' + os.path.join(path_to_mri_images, moving_image) + \
                              ' -p ' + parameter_file + \
                              ' -out ' + path_to_elastix_output + ' > /dev/null'

            # Call elastix
            os.system(elastix_command)

            # # DELETE UNNECESSARY FILES
            # if delete_files:
            #     os.remove(os.path.join(path_to_elastix_output, 'result.0.mhd'))
            #     os.remove(os.path.join(path_to_elastix_output, 'result.0.raw'))

        print("...done! t =", "{:.1f}".format(time.time() - t0), 's\n')


    ####################################################################################################################
    compute_point_transform = True

    if compute_point_transform == True:
        print("Running transformation of input points... ")
        t0 = time.time()

        # Loop over all time frames
        for t in range(n_frames - 1):

            # Assign transformix output
            path_to_transformix_output = os.path.join(path_to_output, 'transformix_result_' + time_frames[t] + '-' + time_frames[t + 1])
            if not os.path.exists(path_to_transformix_output):
                os.makedirs(path_to_transformix_output)

            # Path to input file
            path_to_input_file = os.path.join(path_to_transformix_output, 'inputpoints.txt')

            # Path to transform parameter file
            transform_parameters_file = os.path.join(path_to_output, 'elastix_result_' + time_frames[t] + '-' + time_frames[t + 1])
            transform_parameters_file = transform_parameters_file + '/TransformParameters.0.txt'

            # Flatten the 2D grid to 1D array
            temp_x_points = target_coordinates_x[:, :, t].flatten()
            temp_y_points = target_coordinates_y[:, :, t].flatten()

            # Write a text file
            file = open(path_to_input_file, 'w')    # Overwrite content
            file.write('point\n')                   # Tell elastix that we are dealing with physical points
            file.write(str(im_length) + '\n')       # Number of points
            for x, y in zip(temp_x_points, temp_y_points):
                file.write("{:.6f}".format(x) + '\t' + "{:.6f}".format(y) + '\n')
            file.close()

            # Combine the transformix input command
            transformix_command = path_to_transformix + \
                                  ' -def ' + path_to_input_file + \
                                  ' -out ' + path_to_transformix_output + \
                                  ' -tp ' + transform_parameters_file + ' > /dev/null'

            # Call transformix
            os.system(transformix_command)

            # Name of the output file
            transform_parameters_file = os.path.join(path_to_output, 'elastix_result_' + time_frames[t] + '-' + time_frames[t + 1])
            transform_parameters_file = transform_parameters_file + '/TransformParameters.0.txt'


            path_to_output_file = os.path.join(path_to_output, 'transformix_result_' + time_frames[t] + '-' + time_frames[t + 1])
            path_to_output_file = path_to_output_file + '/outputpoints.txt'

            # Read in the data
            file = open(path_to_output_file, 'r')
            lines = file.readlines()

            # Initialize temporary output
            temp_target_coordinates_x = np.zeros((im_length, 1), dtype="float32")
            temp_target_coordinates_y = np.zeros((im_length, 1), dtype="float32")

            # Find the transformed coordinates
            k = 0
            for line in lines:

                # Get the data from each line
                transformed_points = line.split('\t')
                transformed_points = transformed_points[5]
                transformed_points = transformed_points.split(' ')

                # Assign the x and y coordinate
                temp_target_coordinates_x[k] = float(transformed_points[4])
                temp_target_coordinates_y[k] = float(transformed_points[5])

                # Update the counter
                k += 1

            # Reshape the 1D array to get the original 2D matrix format
            target_coordinates_x[:, :, t + 1] = np.reshape(temp_target_coordinates_x, (im_dir_1, im_dir_2))
            target_coordinates_y[:, :, t + 1] = np.reshape(temp_target_coordinates_y, (im_dir_1, im_dir_2))

        print("...done! t =", "{:.1f}".format(time.time() - t0), 's\n')

    ####################################################################################################################
    # Loop over all time frames
    for t in range(n_frames - 1):

        # Get the relative displacement field
        displacement_field_x[:, :, t + 1] = target_coordinates_x[:, :, t + 1] - target_coordinates_x[:, :, t]
        displacement_field_y[:, :, t + 1] = target_coordinates_y[:, :, t + 1] - target_coordinates_y[:, :, t]

        # Compute the gradient of the displacement field
        gx_x, gx_y = np.gradient(displacement_field_x[:, :, t + 1])
        gy_x, gy_y = np.gradient(displacement_field_y[:, :, t + 1])

        # Add 1 because displacement field is relative: y = x + w(x), so dy/dx = 1 + dw(x)/dx
        gx_x += 1.0
        gy_y += 1.0

        # Apply Gaussian filter to the gradient field
        gaussian_filter(gx_x, sigma=2, output=gx_x)
        gaussian_filter(gx_y, sigma=2, output=gx_y)
        gaussian_filter(gy_x, sigma=2, output=gy_x)
        gaussian_filter(gy_y, sigma=2, output=gy_y)

        # Compute the determinant of the Jacobian for each entry
        jacobian_determinant[:, :, t + 1] = ((gx_x * gy_y) - (gy_x * gx_y))


        # Compute element areas after deformation for each time frame
        quad_areas = cf.uniformRectangularMesh2D(target_coordinates_x[:, :, t + 1],
                                                 target_coordinates_y[:, :, t + 1])
        quad_areas = quad_areas.element_areas.reshape(im_dir_1 - 1, im_dir_2 - 1)

        # Apply Gaussian filter to the areas field
        gaussian_filter(quad_areas, sigma=2, output=quad_areas)

        # Assign areas to the output
        quad_element_areas[:, :, t + 1] = quad_areas


    # Get only the relative change in volume
    # jacobian_determinant -= 1.0
    # quad_element_areas -= 1.0

    # Get the maximum and minimum of jacobian
    jacobian_min = jacobian_determinant.min()
    jacobian_max = jacobian_determinant.max()

    # # Get the maximum and minimum of element areas
    quad_element_areas_min = quad_element_areas.min()
    quad_element_areas_max = quad_element_areas.max()


    ####################################################################################################################


    #
    # # DELETE UNNECESSARY FILES
    # if delete_files:
    #     os.remove(path_to_input_file)
    #     os.remove(path_to_output_file)



    # Downsample the images for plotting
    downsampling = 15
    u_x_all = displacement_field_x[::downsampling, ::downsampling, : ]
    u_y_all = displacement_field_y[::downsampling, ::downsampling, :]
    x_grid_all_target = target_coordinates_x[::downsampling, ::downsampling, :]
    y_grid_all_target = target_coordinates_y[::downsampling, ::downsampling, :]






    # PLOTTING OPTIONS
    plot_deformation_mesh = False
    plot_deformation_vectors = False
    plot_volume_change_quad = True
    plot_jacobian = True




    # PLOTTING OF DEFORMATION MESH
    if plot_deformation_mesh == True:

        line_width = 0.5
        marker_size = 0

        print('\n')
        print('-' * 80)
        print('Plotting Deformation Mesh...')

        # Loop over all time frames
        for t in range(n_frames):

            # Print current time frame
            print('- time frame: ' + time_frames[t])

            # Initialize the figure
            fig, ax = plt.subplots()
            ax.set_xlim(0, im_dir_1 - 1)
            ax.set_ylim(0, im_dir_2 - 1)
            plt.gca().invert_yaxis()
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Z-coordinate')
            ax.set_aspect('equal')

            # Load images in each time frame
            source_im_mri = imageio.imread(os.path.join(path_to_mri_images, 'dynSeq_' + time_frames[t] + '.png'))

            # Show the original image
            img1 = plt.imshow(source_im_mri, cmap=plt.cm.gray, interpolation='none')

            # Get the current deformation values
            x_grid_target = x_grid_all_target[:, :, t]
            y_grid_target = y_grid_all_target[:, :, t]

            # Plot the deformed grid lines
            for x, y in zip(x_grid_target, y_grid_target):
                plt.plot(x, y, '-o', color='red', linewidth=line_width, markersize=marker_size)
            for x, y in zip(np.transpose(x_grid_target), np.transpose(y_grid_target)):
                plt.plot(x, y, '-o', color='red', linewidth=line_width, markersize=marker_size)

            fig.savefig(os.path.join(path_to_output, 'deformation_mesh_' + time_frames[t] + '.png'), dpi=300)
            plt.close(fig)



    # PLOTING OF DEFORMATION VECTORS
    if plot_deformation_vectors == True:

        line_width = 0.5
        marker_size = 0

        print('\n')
        print('-' * 80)
        print('Plotting Displacement Field...')

        # Initialize the figure
        fig, ax = plt.subplots()
        ax.set_xlim(0, im_dir_1 - 1)
        ax.set_ylim(0, im_dir_2 - 1)
        plt.gca().invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Z-coordinate')
        ax.set_aspect('equal')

        # Loop over all time frames
        for t in range(n_frames):
            # Print current time frame
            print('-> time frame: ' + time_frames[t])

            # Get the current deformation values
            u_x = u_x_all[:, :, t]
            u_y = u_y_all[:, :, t]
            x_grid_target = x_grid_all_target[:, :, t] - u_x
            y_grid_target = y_grid_all_target[:, :, t] - u_y

            # Load original mri image
            source_im_mri = imageio.imread(os.path.join(path_to_mri_images, 'dynSeq_' + time_frames[t] + '.png'))

            # Show the original image
            im1 = plt.imshow(source_im_mri, cmap=plt.cm.gray, interpolation='none')

            # Plot the vector field
            cmap = matplotlib.cm.summer
            color = cmap(t / float(n_frames))
            ax.quiver(x_grid_target, y_grid_target, u_x, u_y, angles='xy', scale_units='xy', scale=1, color=color,
                      width=0.002)

            fig.savefig(os.path.join(path_to_output, 'deformation_vectors_' + time_frames[t] + '.png'), dpi=300)

        plt.close(fig)




    # PLOTTING OF VOLUME CHANGE
    if plot_volume_change_quad == True:

        line_width = 1
        marker_size = 1

        print('\n')
        print('-' * 80)
        print('Plotting Relative Volume Change (quadrilateral elements)...')

        # Loop over all time frames
        for t in range(n_frames):
            # Print current time frame
            print('- time frame: ' + time_frames[t])

            # Initialize the figure
            fig, ax = plt.subplots()
            ax.set_xlim(0, im_dir_1 - 1)
            ax.set_ylim(0, im_dir_2 - 1)
            plt.gca().invert_yaxis()
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Z-coordinate')
            ax.set_aspect('equal')

            # Load images in each time frame mri image
            source_im_mri = imageio.imread(os.path.join(path_to_mri_images, 'dynSeq_' + time_frames[t] + '.png'))
            # source_im_edges = imageio.imread(os.path.join(path_to_all_edges, 'dynSeq_edge_all_' + time_frames[t] + '.png'))

            # Get the current deformation values
            coordinates_x = target_coordinates_x[:, :, t]
            coordinates_y = target_coordinates_y[:, :, t]
            area = quad_element_areas[:, :, t]


            img1 = plt.imshow(source_im_mri, cmap=plt.cm.gray, interpolation='none')
            plt.pcolormesh(coordinates_x, coordinates_y, area, shading='flat', cmap='seismic', edgecolors='none', alpha=0.5)

            plt.colorbar(label="<-compression | relative volume deviation | extension->")
            # plt.clim(-quad_element_areas_max, quad_element_areas_max)

            # fig.show()
            fig.savefig(os.path.join(path_to_output, 'volume_deviation_quad_' + time_frames[t] + '.png'), dpi=300)
            plt.close(fig)



    # PLOTING OF DETERMINANT OF JACOBIAN
    if plot_jacobian == True:
        print('\n')
        print('-' * 80)
        print('Plotting Jacobian...')

        # Loop over all time frames
        for t in range(n_frames):
            # Print current time frame
            print('- time frame: ' + time_frames[t])

            # Initialize the figure
            fig, ax = plt.subplots()
            ax.set_xlim(0, im_dir_1 - 1)
            ax.set_ylim(0, im_dir_2 - 1)
            plt.gca().invert_yaxis()
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Z-coordinate')
            ax.set_aspect('equal')

            # Load original mri image
            source_im_mri = imageio.imread(os.path.join(path_to_mri_images, 'dynSeq_' + time_frames[t] + '.png'))

            # Show the Jacobian
            fig, ax = plt.subplots()
            img1 = plt.imshow(source_im_mri, cmap=plt.cm.gray, interpolation='none')
            img2 = plt.imshow(jacobian_determinant[:, :, t].transpose(), cmap='seismic', interpolation='none',
                              alpha=0.6)  # seismic,PiYG,RdYlGn
            plt.colorbar()
            # plt.clim(-jacobian_max, jacobian_max)

            # fig.show()
            fig.savefig(os.path.join(path_to_output, 'jacobian_map_' + time_frames[t] + '.png'), dpi=300)
            plt.close(fig)

