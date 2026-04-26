# Methods to calculate noise parameters for DUT
import skrf as rf
import numpy as np

def Lanes_method(source_sparams, F) -> np.ndarray:
    """Calculate noise parameters using Lanes method.
    
    Parameters:
        source_sparams: List of M skrf.Network objects representing the source S-parameters for each receiver measurement.
        F: 2-D matrix of size (N, M) representing noise factor at the DUT, where N is the number of frequency points.
        
    Returns:
        Np: 2-D matrix of size (N, 4) representing the noise parameters [Rn, Fmin, Gopt, Bopt] at each frequency point.
    """
    # check if all inputs have same number of frequency points
    num_freq_points = source_sparams[0].frequency.size
    for sparam in source_sparams:
        if sparam.frequency.size != num_freq_points:
            raise ValueError("All source S-parameters must have the same number of frequency points.")
    if F.shape[0] != num_freq_points or F.shape[1] != len(source_sparams):
        raise ValueError("F must be a 2-D matrix of size (N, M) where N is the number of frequency points and M is the number of source S-parameters.")
    if len(source_sparams) < 4:
        raise ValueError("At least 4 source S-parameters are required to solve for noise parameters.")
    
    # Rewrite source S11s
    Y_s = np.zeros((num_freq_points, len(source_sparams)))
    G_s = np.zeros((num_freq_points, len(source_sparams)))
    B_s = np.zeros((num_freq_points, len(source_sparams)))
    
    for i in range(len(source_sparams)):
        Y = source_sparams[i].y
        Y_s[:, i] = np.abs(Y[:, 0, 0])
        G_s[:, i] = np.real(Y[:, 0, 0])
        B_s[:, i] = np.imag(Y[:, 0, 0])
    
    A = np.zeros((num_freq_points, len(source_sparams), 4)) # n_f x M measurements x 4 coeffcients
    for i in range(len(source_sparams)):
        A[:, i, 0] = 1
        A[:, i, 1] = Y_s[:, i]**2/G_s[:, i]
        A[:, i, 2] = 1/G_s[:, i]
        A[:, i, 3] = B_s[:, i]/G_s[:, i]
        
    # Solve for noise parameters using least squares
    Np = np.zeros((num_freq_points, 4))
    if len(source_sparams) == 4:
        x = np.linalg.solve(A, F)
    
    if len(source_sparams) > 4:
        x = np.zeros((num_freq_points, 4))
        for f in range(num_freq_points):
            A_f = A[f, :, :]
            F_f = F[f, :]
            x[f, :], _, _, _ = np.linalg.lstsq(A_f, F_f, rcond=None)
            
    # re-parameterize
    a = x[:, 0]
    b = x[:, 1]
    c = x[:, 2]
    d = x[:, 3]
     
    Np[:, 0] = b # Rn
    Np[:, 1] = a + np.sqrt(4*c*b-d**2) # Fmin
    Np[:, 2] = np.sqrt(4*c*b-d**2) / (2*b) # Gopt
    Np[:, 3] = -d / (2*b) # Bopt
    
    return Np