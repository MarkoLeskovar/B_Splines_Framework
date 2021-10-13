import os
import time
import custom_functions as cf


# INFO
# This program loads-in the source image and the transform parameters and then
# preforms elastix transformation to obtain the transformed target image.
# Transformation is preformed via transformix.
if __name__ == "__main__":

    # Elastix and Tranformix system paths
    path_to_elastix = '/home/mleskova/install_dir/elastix-5.0.1-linux/install/bin/elastix'
    path_to_transformix = '/home/mleskova/install_dir/elastix-5.0.1-linux/install/bin/transformix'

    # Path to data relative to current working directory
    path_to_input = 'output_dummy_data_2D/square'
    path_to_output = 'output_dummy_data_2D/square'

    # Get the path to working directory, input and output
    path_to_working_directory = os.getcwd()
    path_to_input = os.path.join(path_to_working_directory, path_to_input)
    path_to_output = os.path.join(path_to_working_directory, path_to_output)

    # Load path to images, output and parameter file
    input_image = '01_grid.png'
    output_image = '02_grid.png'

    # path_to_bone_output = os.path.join(path_to_output, 'elastix_reg_all')
    path_to_bone_output = path_to_output

    # Load the parameter file
    # WARNING!
    # both the center of rotation and image origin should be at [0,0] as they are not included in the equations
    parameter_file = 'TransformParameters.0.txt'


    # Combine the elastix input command
    transformix_command = path_to_transformix + \
                          ' -in ' + os.path.join(path_to_input, input_image) + \
                          ' -out '+ path_to_output + \
                          ' -tp ' + os.path.join(path_to_input, parameter_file) + \
                          ' > log.txt'

    # Run  elastix in command line
    print("Running transformix image transformation... " )
    t0 = time.time()

    # Call elastix
    # print(transformix_command)
    os.system(transformix_command)

    print("...done! t =", "{:.1f}".format(time.time() - t0), 's\n')


    # Read in the mhd image and save it as png
    result_im = cf.mhd_image(os.path.join(path_to_output, 'result.0.mhd'))
    result_im.convertToPNG(os.path.join(path_to_output, output_image))


    # FILE OPTIONS
    delete_point_files = False

    # DELETE UNNECESSARY FILES
    if delete_point_files:
        os.remove(os.path.join(path_to_output, 'result.0.mhd'))
        os.remove(os.path.join(path_to_output, 'result.0.raw'))
