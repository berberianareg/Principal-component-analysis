"""Principal component analysis for dimensionality reduction.

Notes
-----
  This script is version v0. It provides the base for all subsequent
  iterations of the project.

Requirements
------------
  See "requirements.txt"
"""

#%% import libraries and modules
import numpy as np  
import matplotlib.pyplot as plt
import os

#%% figure parameters
plt.rcParams['figure.figsize'] = (5,5)
plt.rcParams['font.size']= 14

#%% build PCA class
class PCA:
    """Pincipal component analysis class."""
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
    def make_data(self):
        """Create the original data."""
        # set random seed (default: 42)
        np.random.seed(42)
        # set x-coordinates (default: mu=1; sd=0.5)
        x = np.random.normal(1, 0.5, (self.num_samples, 1))
        # set y-coordinates (default: mu=1; sd=0.5)
        y = x + np.random.normal(1, 0.5, (self.num_samples, 1))
        # concatenate x and y coordinates
        data = np.hstack((x, y))
        return data
    
    def transform_data(self, data):
        """Transform the original data."""
        # subtract mean from original data
        data_transformed = data - data.mean(axis=0)
        return data_transformed
    
    def get_covariance_matrix(self, data_transformed):
        """Compute the covariance matrix."""
        # compute covariance matrix
        covariance_matrix = np.dot(data_transformed.T, data_transformed) / (self.num_samples-1)
        # round to 1 decimal place
        covariance_matrix = covariance_matrix.round(1)
        return covariance_matrix
    
    def normalize_eigenvectors(self, eigenvectors):
        """Normalize eigenvectors."""
        # compute norm of eigenvectors
        norm = np.sqrt(np.sum(np.square(eigenvectors), axis=0))
        # normalize eigenvectors
        eigenvectors_normalized = eigenvectors / norm
        return eigenvectors_normalized
    
#%% instantiate PCA class
pca = PCA()

#%% make data
data = pca.make_data()

#%% transform data
data_transformed = pca.transform_data(data)

#%% get covariance matrix
covariance_matrix = pca.get_covariance_matrix(data_transformed)
print('covariance_matrix = {}'.format(covariance_matrix))

#%% compute eigenvectors and corresponding eigenvalues of the covariance matrix using numpy
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
print('eigenvector v = {} has corresponding eigenvalue \u03BB = {}'.format(tuple(eigenvectors[0].round(3)), eigenvalues[0]))
print('eigenvector v = {} has corresponding eigenvalue \u03BB = {}'.format(tuple(eigenvectors[1].round(3)), eigenvalues[1]))

#%% results from calculations by hand (see README.md file)
eigenvectors_found = np.array([[-2, 1], [1, 2]])
eigenvectors_normalized = pca.normalize_eigenvectors(eigenvectors_found)

# check if hand calculations are equal to numpy calculations
assert (eigenvectors_normalized == eigenvectors).all()

#%% plot figures

cwd = os.getcwd()                                                               # get current working directory
fileName = 'images'                                                             # specify filename

# filepath and directory specifications
if os.path.exists(os.path.join(cwd, fileName)) == False:                        # if path does not exist
    os.makedirs(fileName)                                                       # create directory with specified filename
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory
else:
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory

x = data[:, 0]
y = data[:, 1]

fig, ax = plt.subplots()
ax.scatter(x, y, s=5, color='red')
ax.quiver(x.mean(), y.mean(), eigenvectors[0, 0] * np.sqrt(eigenvalues[0]), eigenvectors[1, 0] * np.sqrt(eigenvalues[0]), scale_units='xy', scale=1)
ax.quiver(x.mean(), y.mean(), eigenvectors[0, 1] * np.sqrt(eigenvalues[1]), eigenvectors[1, 1] * np.sqrt(eigenvalues[1]), scale_units='xy', scale=1)
ax.set_title('principal component analysis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_1'))
