import numpy as np
import scipy as sc
import nibabel as nib
import SimpleITK as sitk
from PIL import Image

# mhd image class
class mhd_image:
    def __init__(self, filename):

        # Reads the image using SimpleITK
        data = sitk.ReadImage(filename)

        # Convert the image to a numpy array first and then shuffle the dimensions to get axis in the order x,y,z
        self.ct_scan = np.array(sitk.GetArrayFromImage(data)).transpose()

        # Image dimensions
        self.shape = np.array(self.ct_scan.shape)

        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        # TODO: check this on non-uniform origin
        self.origin = np.array(data.GetOrigin())

        # Read the spacing along each dimension
        # TODO: check this on non-uniform spacing
        self.spacing = np.array(data.GetSpacing())

    # Function that save the image to a file
    def convertToPNG(self,output_file):
        temp_im = self.ct_scan.astype(np.uint8)
        temp_im = Image.fromarray(temp_im.transpose())
        temp_im = temp_im.convert("L")
        temp_im.save(output_file)



    # TODO: create a new constructor to create image from array and
    #  another constructor so that I have the same type of images

# Function to calculate distance transform
def computeDistanceToMask(bone_mask):

    # Invert the values
    bone_mask = np.subtract(np.max(bone_mask), bone_mask)

    # Distance transform
    return sc.ndimage.distance_transform_edt(bone_mask)

# Function to calculate weight of each pixel
def computeComponentWeightingFunction(data):

    # Weighting function 1
    # return 1.0 / (1.0 + computeDistanceToMask(data) ** 2)

    # Weighting function 2
    # alpha = 0.5
    # beta = 2
    # return 1.0 / (1.0 + alpha * computeDistanceToMask(data) ** beta)

    # Weighting function 3
    # 0.4 ... 0.8
    # gamma = 0.4
    # return 2.0 / (1.0 + np.exp(gamma * computeDistanceToMask(data)))

    # Weighting function 4
    return 1.0 / (1.0 + 0.5 * computeDistanceToMask(data) ** 2)

# Function that reads-in the elastix output file and outputs the transformation parameters
def getTransformParameters_2D(parameter_file):

    # Initialize the output
    parameters = np.empty(5)

    # Open the parameter file
    file = open(parameter_file, 'r')
    lines = file.readlines()

    # Find the transform parameters
    for line in lines:
        line_split = line.split(" ")

        if line_split[0] == "(TransformParameters":
            parameters[0] = float(line_split[1])
            parameters[1] = float(line_split[2])
            parameters[2] = float(line_split[3].split(")")[0])

            # print("Transform parameters:", parameters[0], parameters[1], parameters[2])
            break

    # Find the centre of rotation
    for line in lines:
        line_split = line.split(" ")

        if line_split[0] == "(CenterOfRotationPoint":
            parameters[3] = float(line_split[1])
            parameters[4] = float(line_split[2].split(")")[0])

            # print("Center of rotation point:", parameters[3], parameters[4], "\n")
            break

    return parameters

# Function that reads-in the elastix output file and outputs the transformation parameters
def getTransformParameters_3D(parameter_file):

    # Initialize the output
    parameters = np.empty(9)

    # Open the parameter file
    file = open(parameter_file, 'r')
    lines = file.readlines()

    # Find the transform parameters
    for line in lines:
        line_split = line.split(" ")

        if line_split[0] == "(TransformParameters":
            parameters[0] = float(line_split[1])
            parameters[1] = float(line_split[2])
            parameters[2] = float(line_split[3])
            parameters[3] = float(line_split[4])
            parameters[4] = float(line_split[5])
            parameters[5] = float(line_split[6].split(")")[0])

            # print("Transform parameters:", parameters[:6])
            break

    # Find the centre of rotation
    for line in lines:
        line_split = line.split(" ")

        if line_split[0] == "(CenterOfRotationPoint":
            parameters[6] = float(line_split[1])
            parameters[7] = float(line_split[2])
            parameters[8] = float(line_split[3].split(")")[0])

            # print("Center of rotation point:", parameters[6], parameters[7], parameters[7], "\n")
            break

    return parameters


# Function to get the transformation matrix
def getTransformationMatrix_2D(parameter_file):

    # Get transformation parameters
    parameters = getTransformParameters_2D(parameter_file)

    # Define the transformation  matrix
    transformation_matrix = np.array([[ np.cos(parameters[0]), -1.0 * np.sin(parameters[0]), parameters[1] ],
                                      [ np.sin(parameters[0]), np.cos(parameters[0])       , parameters[2] ],
                                      [ 0.0                  , 0.0                         , 1.0           ]])

    return transformation_matrix

# Function to get the transformation matrix
def getTransformationMatrix_3D(parameter_file):

    # Get transformation parameters
    parameters = getTransformParameters_3D(parameter_file)

    # Get the constants from the parameter file
    # TODO: Check this is errors occur
    cosA = np.cos(parameters[2]) # around z-axis
    cosB = np.cos(parameters[1]) # around y-axis
    cosC = np.cos(parameters[0]) # around x-axis

    sinA = np.sin(parameters[2]) # around z-axis
    sinB = np.sin(parameters[1]) # around y-axis
    sinC = np.sin(parameters[0]) # around x-axis

    tx = parameters[3]
    ty = parameters[4]
    tz = parameters[5]

    # Define the transformation  matrix
    transformation_matrix = np.array([[ cosA*cosB, cosA*sinB*sinC - sinA*cosC, cosA*sinB*cosC + sinA*sinC, tx ],
                                      [ sinA*cosB, sinA*sinB*sinC + cosA*cosC, sinA*sinB*cosC - cosA*sinC, ty ],
                                      [ -sinB    , cosB*sinC                 , cosB*cosC                 , tz ],
                                      [ 0        , 0                         , 0                         , 1  ]])

    return transformation_matrix

# Compute matrix logartihm using the eigendecomposition method
def matrixLogarithm(matrix):

    D, P = np.linalg.eig(matrix)
    P_inv = np.linalg.inv(P)
    D = np.diag(np.log(D))

    return np.dot(P,np.dot(D,P_inv))

# Compute matrix exponential using the eigendecomposition method
def matrix_exponential(matrix):

    D, P = np.linalg.eig(matrix)
    P_inv = np.linalg.inv(P)
    D = np.diag(np.exp(D))

    return np.dot(P, np.dot(D, P_inv))


# Define a Quadrilateral element class to compute area after deformation
class quad2D:
    def __init__(self,p_1,p_2,p_3,p_4):
        self.p1 = p_1
        self.p2 = p_2
        self.p3 = p_3
        self.p4 = p_4

        # Compute area
        self.area = 0.5 * abs( (self.p1[0]*self.p2[1] + self.p2[0]*self.p3[1] + self.p3[0]*self.p4[1] + self.p4[0]*self.p1[1]) -
                            (self.p2[0]*self.p1[1] + self.p3[0]*self.p2[1] + self.p4[0]*self.p3[1] + self.p1[0]*self.p4[1]) )

# Function that calculates element areas in 2D
def computeElementAreas2D(coordinates_x, coordinates_y):

    # Get image shape
    shape = np.shape(coordinates_x)

    # Initialize the output
    areas = np.zeros((shape[0] - 1, shape[1] - 1), dtype="float32")

    # Mesh the grid with Quadrilaterals
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):

            # Get coordinates
            p_1 = [coordinates_x[i    , j    ], coordinates_y[i    , j    ]]
            p_2 = [coordinates_x[i    , j + 1], coordinates_y[i    , j + 1]]
            p_3 = [coordinates_x[i + 1, j + 1], coordinates_y[i + 1, j + 1]]
            p_4 = [coordinates_x[i + 1, j    ], coordinates_y[i + 1, j    ]]

            # Create and element
            areas[i, j] = quad2D(p_1, p_2, p_3, p_4).area

    return areas


# Function that creates an uniform rectangular mesh from a pair of x and y grid coordinates
class uniformRectangularMesh2D:

    # Default constructor
    def __init__(self, grid_coordinates_x, grid_coordinates_y):

        # Initialize mesh shape
        self.shape = np.shape(grid_coordinates_x)

        # Ge the number of nodes and elements from input arrays
        self.num_nodes = self.shape[0] * self.shape[1]
        self.num_elements = (self.shape[0] - 1) * (self.shape[1] - 1)

        # Initialize nodes and elements id arrays
        self.nodes = np.zeros((self.num_nodes, 2), dtype='float32')
        self.elements = np.zeros((self.num_elements, 4), dtype='int')

        # Create an array of nodes that contains the coordinates of each node
        self.nodes[:, 0] = grid_coordinates_x.flatten()
        self.nodes[:, 1] = grid_coordinates_y.flatten()

        # Create a list of elements
        k = -1
        l = 0
        for n in range(self.shape[0] - 1):  # width
            k += 1
            for m in range(self.shape[1] - 1):  # height

                # Get node indices
                self.elements[l, :] = [k, k + 1, k + self.shape[1], k + self.shape[1] + 1]
                k += 1
                l += 1

        # Initialize a list of edges that contains node connections
        self.edges = np.empty((self.num_elements * 4, 2), dtype='int')

        # Create a list of edges to plot
        k = 0
        for n in range(self.num_elements):

            # Get the node ids from element list
            node_ids = self.elements[n, :]

            # Get the edges for each element
            self.edges[k    , :] = [node_ids[0], node_ids[1]]
            self.edges[k + 1, :] = [node_ids[1], node_ids[3]]
            self.edges[k + 2, :] = [node_ids[3], node_ids[2]]
            self.edges[k + 3, :] = [node_ids[2], node_ids[0]]
            k += 4

        # Keep only unique edges
        self.edges = np.unique(np.sort(self.edges, axis=1), axis=0)
        self.num_edges = self.edges.shape[0]

        # Calculate the area of each mesh element
        self.element_areas = computeElementAreas2D(grid_coordinates_x, grid_coordinates_y)

    # Function to return the mesh element
    def getElementIndex(self, i, j):
        return j + i * (self.shape[1]-1)
