import numpy as np
from scipy.linalg import eig, norm, eigvals
from numpy import linalg as la
from numalgsolve.polynomial import MultiCheb, MultiPower

from matplotlib import pyplot as plt
def solve(poly, method = 'mult', eigvals=True):
    """Finds the zeros of a 1-D polynomial.

    Parameters
    ----------
    poly : Polynomial
        The polynomial to find the roots of.

    method : str
        'mult' will use the multiplicaiton matrix technique.
        'div' will use the division matrix technique.
        Defaults to 'mult'

    Returns
    -------
    one_dimensional_solve : numpy array
        An array of the zeros.
    """
    if method != 'mult' and method != 'div':
        raise ValueError('method must be mult or div!')

    if type(poly) == MultiPower:
        size = len(poly.coeff)
        coeff = np.trim_zeros(poly.coeff)
        zeros = np.zeros(size - len(coeff), dtype = 'complex')
        if method == 'mult':
            return np.hstack((zeros,multPower(coeff, eigvals)))
        else:
            return np.hstack((zeros,divPower(coeff, eigvals)))
    else:
        if method == 'mult':
            return multCheb(poly.coeff, eigvals)
        else:
            return divCheb(poly.coeff, eigvals)

def multPower(coeffs, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using a multiplication matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[n::n+1]
    bot[...] = 1#*coeffs[-1]
    matrix[:, -1] -= coeffs[:-1]/coeffs[-1]

    if eigvals:
        zeros = la.eigvals(matrix)
        return zeros#/coeffs[-1]
    else:
        vals,vecs = eig(matrix.T)
        return vecs[1,:]/vecs[0,:]

def logplot(vals):
    z = np.sign(vals)*np.log(np.abs(vals)+1)
    plt.imshow(z)
    plt.colorbar()
    plt.show()

def multnPower(coeffs, mult_n, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using a multiplication matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.
    mult_n : int
        The power of x for the multiplication matrix.
    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.eye(n, dtype=coeffs.dtype)

    xn = -(coeffs[:-1]/coeffs[-1]).reshape((n,1))
    for i in range(mult_n):
        remainder = matrix[-1,-i:].copy()
        matrix[-1] = 0
        matrix = np.roll(matrix, 1, axis=0)
        matrix[:,-i:] += remainder*xn

    plt.imshow(np.abs(matrix))
    plt.title(f"{mult_n}")
    plt.colorbar()
    plt.show()


    if eigvals:
        zeros = la.eigvals(np.rot90(matrix,2))**(1/mult_n)
        return zeros
    else:
        vals,vecs = eig(np.rot90(matrix,2), left=True, right=False)
        print(vecs[-5:,0])
        return np.conjugate(vecs[-2,:]/vecs[-1,:])
        # vals,vecs = eig(np.rot90(matrix,2), left=True, right=False)
        # return np.conjugate(vecs[0,:]/vecs[1,:])

        # idx = np.argmax(np.abs(vecs), axis=0)
        # return np.conjugate(vecs[idx-1,:]/vecs[idx,:])
        # vals,vecs = eig(matrix, left=True, right=False)
        # return vecs[0,:]/vecs[1,:]

def multPowerR(coeffs, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using a multiplication matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[n::n+1]
    bot[...] = 1
    matrix[:, -1] -= coeffs[:-1]/coeffs[-1]
    if eigvals:
        zeros = la.eigvals(np.rot90(matrix,2))
        return zeros
    else:
        from scipy.linalg import eig
        vals,vecs= eig(np.rot90(matrix,2), left=True, right=False)
        return np.conjugate(vecs[0,:]/vecs[1,:])

def divPower(coeffs, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using a division matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[1::n+1]
    bot[...] = 1
    matrix[:, 0] -= coeffs[1:]/coeffs[0]
    # print(matrix)
    if eigvals:
        zeros = 1/la.eigvals(matrix)
        return zeros
    else:
        from scipy.linalg import eig
        vals,vecs = eig(matrix, left=True, right=False)
        return np.conjugate(vecs[1,:]/vecs[0,:])

def div2Power(coeffs, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using a twice division matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[2:n*(n-2):n+1]
    bot[...] = 1*coeffs[0]
    # print(matrix)
    # print(coeffs)
    matrix[:, 1] -= coeffs[1:]#/coeffs[0]
    matrix[:, 0] -= coeffs[1]*matrix[:, 1]/coeffs[0]
    matrix[:-1, 0] -= coeffs[2:]#/coeffs[0]
    # print(matrix)

    from matplotlib import pyplot as plt
    # plt.imshow(matrix.real)
    # plt.colorbar()
    # plt.show()
    if eigvals:
        zeros = 1/la.eigvals(matrix)
        return zeros/coeffs[0]
    else:
        from scipy.linalg import eig
        vals,vecs = eig(matrix, left=True, right=False)
        return np.conjugate(vecs[1,:]/vecs[0,:])

def divnPower(coeffs, div_n, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using an nth division matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.
    div_n : int
        Use x^(-n) for functions

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    assert div_n > 0
    n = len(coeffs) - 1
    matrix = np.eye(n, dtype=coeffs.dtype)
    #bot = matrix.reshape(-1)[2:n*(n-2):n+1]
    #bot[...] = 1*coeffs[0]
    # print(matrix)
    # print(coeffs)
    xinv = (-coeffs[1:]/coeffs[0]).reshape((n,1))
    # print('xinv',xinv)
    remainder = np.zeros(n)
    for i in range(div_n):
        remainder = matrix[0, :i+1].copy()
        matrix[0] = 0
        matrix = np.roll(matrix, -1, axis=0)
        matrix[:,:i+1] += remainder*xinv
        # print("i\n", matrix)
    # matrix[:, 1] -= coeffs[1:]#/coeffs[0]
    # matrix[:, 0] -= coeffs[1]*matrix[:, 1]/coeffs[0]
    # matrix[:-1, 0] -= coeffs[2:]#/coeffs[0]
    # print(matrix)

    from matplotlib import pyplot as plt
    # plt.imshow(matrix.real)
    # plt.colorbar()
    # plt.show()
    if eigvals:
        zeros = la.eigvals(matrix)**(-1/div_n)
        return zeros
    else:
        from scipy.linalg import eig
        vals,vecs = eig(matrix, left=True, right=False)
        return np.conjugate(vecs[1,:]/vecs[0,:])

def plusPower(coeffs, one_n, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using an nth division matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.
    div_n : int
        Use x^(-n) for functions

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    dMatrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = dMatrix.reshape(-1)[1::n+1]
    bot[...] = 1
    dMatrix[:, 0] -= coeffs[1:]/coeffs[0]

    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[n::n+1]
    bot[...] = 1#*coeffs[-1]
    matrix[:, -1] -= coeffs[:-1]/coeffs[-1]

    matrix += dMatrix

    from matplotlib import pyplot as plt
    logplot(np.abs(matrix))

    # zeros = x + 1/x
    # x^2 - zeros*z + 1 = 0
    if eigvals:
        zeros = la.eigvals(matrix)
        zeros = (zeros - np.sqrt(zeros**2 - 4))/(2)
        zeros = np.append(zeros, (zeros + np.sqrt(zeros**2 - 4))/(2))
        return zeros
    else:
        from scipy.linalg import eig
        vals,vecs = eig(np.rot90(matrix,2), left=True, right=False)
        return np.conjugate(vecs[1,:]/vecs[0,:])

def multCheb(coeffs, eigvals=True):
    """Finds the zeros of a 1-D chebyshev polynomial using a multiplication matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    mMatrix = np.zeros((n,n), dtype=coeffs.dtype)
    mMatrix[1][0] = 1
    bot = mMatrix.reshape(-1)[1::n+1]
    bot[...] = 1/2
    bot = mMatrix.reshape(-1)[2*n+1::n+1]
    bot[...] = 1/2
    #print(coeffs[-1])
    mMatrix[:,-1] -= .5*coeffs[:-1]/coeffs[-1]
    if eigvals:
        zeros = la.eigvals(mMatrix)
        return zeros
    else:
        vals,vecs = eig(mMatrix.T)
        return vecs[1,:]/vecs[0,:]

def multChebR(coeffs, eigvals=True):
    """Finds the zeros of a 1-D chebyshev polynomial using a multiplication matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    mMatrix = np.zeros((n,n), dtype=coeffs.dtype)
    mMatrix[1][0] = 1
    bot = mMatrix.reshape(-1)[1::n+1]
    bot[...] = 1/2
    bot = mMatrix.reshape(-1)[2*n+1::n+1]
    bot[...] = 1/2
    #print(coeffs[-1])
    mMatrix[:,-1] -= .5*coeffs[:-1]/coeffs[-1]
    if eigvals:
        zeros = la.eigvals(np.rot90(mMatrix,2))
        return zeros
    else:
        vals,vecs = eig(np.rot90(mMatrix,2))
        plt.imshow(vecs.real)
        plt.colorbar()
        plt.show()
        return vecs[0,:]/vecs[1,:]


def getXinv(coeff):
    n = len(coeff)-1
    curr = coeff.copy()
    xinv = np.zeros(n, dtype=coeff.dtype)
    for i in range(1,n)[::-1]:
        val = -curr[i+1]
        # curr[i+1] += val
        curr[i-1] += val
        xinv[i]+=2*val
    temp = -curr[1]
    # curr[1]+=temp
    xinv[0]+=temp
    #xinv/=curr[0]
    return xinv,curr[0]


def divCheb(coeffs, eigvals=True):
    """Finds the zeros of a 1-D chebyshev polynomial using a division matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    xinv,divisor = getXinv(coeffs)
    n = len(coeffs)-1

    dMatrix = np.zeros((n,n), dtype=coeffs.dtype)

    sign = 1
    for col in range(1,n,2):
        bot = dMatrix.reshape(-1)[col:(n-col)*n:n+1]
        bot[...] = 2*sign
        sign *= -1
    dMatrix[0]/=2

    if abs(divisor) > 1:
        xinv/=divisor
    else:
        dMatrix*=divisor

    sign = 1
    for col in range(0,n,2):
        dMatrix[:,col]+=xinv*sign
        sign*=-1

    if eigvals:
        zerosD = 1/la.eigvals(dMatrix.T)

    else:
        vals,vecs = eig(dMatrix, left=True,right=False)
        zerosD = np.conjugate(vecs[1,:]/vecs[0,:])
        return zerosD

    #print(divisor)

    if abs(divisor) > 1:
        return zerosD
    else:
        return zerosD*divisor

def div2Cheb(coeffs, eigvals=True):
    """Finds the zeros of a 1-D chebyshev polynomial using a division of x^(-2) matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    xinv,divisor = getXinv(coeffs)
    x2inv,divisor2 = getXinv(-xinv)

    x2inv = np.append(x2inv, [0])
    if abs(divisor) > 1:
        xinv /= divisor
        x2inv -= divisor2*xinv
        x2inv/=divisor
    else:
        x2inv -= divisor2*xinv/divisor

    n = len(coeffs)-1

    dMatrix = np.zeros((n,n), dtype=coeffs.dtype)

    sign = 1
    val = 4
    for col in range(2,n,2):
        bot = dMatrix.reshape(-1)[col:(n-col)*n:n+1]
        bot[...] = val*sign
        val += 4
        sign *= -1
    dMatrix[0]/=2

    if abs(divisor) > 1:
        pass
        # xinv/=divisor
    else:
        dMatrix*=divisor

    sign = 1
    val = 1
    for col in range(1,n,2):
        dMatrix[:,col]+=xinv*sign*val
        val += 2
        sign*=-1

    sign = 1
    for col in range(0,n,2):
        dMatrix[:,col]+= x2inv*sign
        sign*=-1

    if eigvals:
        zerosD = 1/la.eigvals(dMatrix)

    else:
        vals,vecs = eig(dMatrix, left=True,right=False)
        # vals,vecs = eig(dMatrix.T, left=False,right=True)
        zerosD = np.conjugate(vecs[1,:]/vecs[0,:])
        # zerosD = vecs[1,:]/vecs[0,:]
        return zerosD

    if abs(divisor) > 1:
        pass
        return zerosD**(1/2)
    else:
        zerosD *= divisor
        return zerosD**(1/2)
