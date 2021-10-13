import os
import time

# INFO
# This program loads-in the source and the target images and preforms elastix transformation
# from time frame 01 (source) to time frame 02 (target). Registration is preformed via elastix.
if __name__ == "__main__":

    # Elastix and Tranformix system paths
    path_to_elastix = '/home/mleskova/install_dir/elastix-5.0.1-linux/install/bin/elastix'
    path_to_transformix = '/home/mleskova/install_dir/elastix-5.0.1-linux/install/bin/transformix'

    # Path to data relative to current working directory
    path_to_input = 'dummy_data_2D/translation'
    path_to_output = 'grid_visualization'

    # Get the path to working directory, input and output
    path_to_working_directory = os.getcwd()
    path_to_input = os.path.join(path_to_working_directory, path_to_input)
    path_to_output = os.path.join(path_to_working_directory, path_to_output)

    # Load path to images, output and parameter file
    source_image = '01_dummy_all.png'
    target_image = '02_dummy_all.png'

    # path_to_bone_output = os.path.join(path_to_output, 'elastix_reg_all')
    path_to_bone_output = path_to_output

    # Load the parameter file
    # WARNING!
    # both the center of rotation and image origin should be at [0,0] as they are not included in the equations
    parameter_file = 'parameters_BSpline_2.txt'

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
        print("Running elastix registration... " )
        t0 = time.time()

        # Call elastix
        # print(elastix_command)
        os.system(elastix_command)

        print("...done! t =", "{:.1f}".format(time.time() - t0), 's\n')

    else:
        print("Output folder already exists! -> ", path_to_bone_output)