import numpy as np
from numalgsolve.polynomial import MultiCheb, MultiPower
from numalgsolve.OneDimension import multPowerR, solve as oneDsolve, div2Power, div2Cheb,divnPower,multnPower, multChebR, plusPower
from numalgsolve.polyroots import solve
from numpy.polynomial.polynomial import polyfromroots, polyroots
from numpy.polynomial.chebyshev import chebfromroots, chebroots
from matplotlib import pyplot as plt
import argparse
import warnings

def choose_points(num_points, radius):
    """Create random complex numbers with a maximum modulus.
    Parameters
    ----------
    num_points : number of points
    radius : maximum radius

    Returns
    -------
    array of random numbers

    """
    pass


def create_roots_graph(args, results):
    nrows = len(results)
    ncols = len(next(iter(results.values())))
    if not args.coeffs: ncols-=1  #get sub dictionary length, but ignore the roots item

    plt.figure(figsize=(2*ncols,6))
    for i,(radius, sub_results) in enumerate(results.items()):
        radius = float(radius)
        if not args.coeffs:
            roots = sub_results['roots']
            del sub_results['roots']
        for j,(method, roots_approx) in enumerate(sub_results.items()):
            ax = plt.subplot(nrows,ncols,i*ncols+(j+1))
            if args.dimension == 1:
                if not args.coeffs: plt.plot(roots.real, roots.imag, 'r+', ms = 7)
                plt.plot(roots_approx.real, roots_approx.imag, 'bo', ms=3)
            else:
                if not args.coeffs: plt.plot(roots[:,0], roots[:,1], 'r+', ms = 7)
                plt.plot(roots_approx[:,0], roots_approx[:,1], 'bo', ms=3)
            r = 1
            plt.xlim(-r,r)#-radius,radius)
            plt.ylim(-r,r)#-radius,radius)
            if j > 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('imag')
            if i < nrows-1:
                ax.set_xticklabels([])
            else:
                plt.xlabel('real')
            if i==0:plt.title(method)

    plt.tight_layout()
    plt.show()

def create_stability_graph():
    raise NotImplementedError()

def logplot(vals):
    z = np.sign(vals)*np.log(np.abs(vals)+1)
    plt.plot(z)
    plt.show()


def run_one_dimension(args, radius):
    num_points = args.num_points
    eps = args.eps
    power = args.power
    real = args.real
    eigvals = True if args.eig == 'val' else False
    by_coeffs = args.coeffs

    res = {}
    if by_coeffs:
        coeffs = (np.random.random(num_points+1)*2 - 1)*radius
        powerpoly = MultiPower(coeffs)
        chebpoly = MultiCheb(coeffs)
    else:
        r = np.random.random(num_points)*radius + eps
        angles = 2*np.pi*np.random.random(num_points)
        if power and not real:
            roots = r*np.exp(angles*1j)
        else:
            roots = 2*r-radius

        # ['roots#{:.2f}'.format(radius)] = roots
        res = {'roots':roots}

        powerpoly = MultiPower(polyfromroots(roots))
        chebpoly = MultiCheb(chebfromroots(roots))
    # plt.subplot(121)
    # plt.hist(np.abs(chebpoly[0](roots)))
    # plt.title("cheb eval at roots")
    # plt.subplot(122)
    # plt.hist(np.abs(powerpoly[0](roots)))
    # plt.title("power eval at roots")
    # plt.show()

    # print('chebpoly:\n',chebpoly[0].coeff)
    # logplot(powerpoly[0].coeff)
    # logplot(chebpoly[0].coeff)

    # res['mult power']  = solve(powerpoly, 'mult')
    # res['multR power']  = multPowerR(powerpoly[0].coeff)
    if power:
        res['multR power']  = multPowerR(powerpoly.coeff, eigvals)
        # res['div power']   = oneDsolve(powerpoly, 'div', eigvals)
        # res['div2 power']   = div2Power(powerpoly.coeff, eigvals)
        res['div1 power']   = divnPower(powerpoly.coeff, 1, eigvals)
        res['div2 power']   = divnPower(powerpoly.coeff, 2, eigvals)
        res['div3 power']   = divnPower(powerpoly.coeff, 3, eigvals)
        res['div4 power']   = divnPower(powerpoly.coeff, 4, eigvals)
        res['div5 power']   = divnPower(powerpoly.coeff, 5, eigvals)
        res['plus power']   = plusPower(powerpoly.coeff, eigvals)

        # res['mult1 power']   = multnPower(powerpoly.coeff, 1, eigvals)
        # res['mult2 power']   = multnPower(powerpoly.coeff, 2, eigvals)
        # res['mult3 power']   = multnPower(powerpoly.coeff, 3, eigvals)
        # res['mult4 power']   = multnPower(powerpoly.coeff, 4, eigvals)
        # res['mult5 power']   = multnPower(powerpoly.coeff, 5, eigvals)
        res['numpy power'] = polyroots(powerpoly.coeff)
    else:
        res['mult cheb']   = oneDsolve(chebpoly, 'mult', eigvals)
        res['multR cheb']   = multChebR(chebpoly.coeff, eigvals)
        res['div cheb']    = oneDsolve(chebpoly, 'div', eigvals)
        res['div2 cheb']    = div2Cheb(chebpoly.coeff, eigvals)
        res['numpy cheb']  = chebroots(chebpoly.coeff)

    # plt.subplot(121)
    # plt.hist(np.abs(powerpoly[0](res['multR power'])))
    # plt.title("multR eval at roots")
    # plt.subplot(122)
    # plt.hist(np.abs(powerpoly[0](res['div2 cheb'])))
    # plt.title("div2 eval at roots")
    # plt.show()
    return res

def run_n_dimension(args, radius):
    num_points = args.num_points
    eps = args.eps
    power = args.power
    real = args.real
    eigvals = True if args.eig == 'val' else False
    by_coeffs = args.coeffs
    dim = args.dimension

    res = {}
    powerpolys = []
    chebpolys = []
    if by_coeffs:
        for i in range(dim):
            from numalgsolve.polynomial import getPoly
            powerpolys.append(getPoly(dim, num_points, power=True))
            chebpolys.append(getPoly(dim, num_points, power=False))
    else:
        r = np.random.random((num_points, dim))*radius + eps
        roots = 2*r-radius

        from itertools import product
        res = {'roots':np.array(list(product(*np.rot90(roots))))}
        print(res['roots'])

        for i in range(dim):
            coeffs = np.zeros((num_points+1,)*dim)
            idx = [slice(None),]*dim
            idx[i] = 0

            coeffs[tuple(idx)] = polyfromroots(roots[:,i])
            powerpolys.append(MultiPower(coeffs))

            coeffs[tuple(idx)] = chebfromroots(roots[:,i])
            chebpolys.append(MultiCheb(coeffs))

    # if power:
    res['mult power'] = solve(powerpolys, 'mult')
    res['div power']  = solve(powerpolys, 'div')
    # else:
    res['mult cheb']  = solve(chebpolys, 'mult')
    res['div cheb']   = solve(chebpolys, 'div')
    print(type(res['mult cheb']))

    return res

def run_roots_testing(args):

    shrink_factor = 0.7
    radius = args.radius
    dim = args.dimension
    results = {}

    for _ in range(3):
        r_trunc = '{:.2f}'.format(radius)
        if dim == 1:
            results[r_trunc] = run_one_dimension(args, radius)
        else:
            results[r_trunc] = run_n_dimension(args, radius)

        radius *= shrink_factor
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Stability Test Options")
    parser.add_argument('-d', '--dimension', type=int, default=1, help='Polynomial dimension')
    parser.add_argument('-n', '--num_points', type=int, default=50, help='Number of complex roots, minimum of 2')
    parser.add_argument('--real', action='store_true', help='Use just real points')
    parser.add_argument('-r', '--radius', type=float, default=1, help='The largest radius for the points')
    parser.add_argument('-e', '--eps', type=float, default=1e-8, help='Number of complex roots, minimum of 2')
    parser.add_argument('-W', '--nowarn', action='store_true', help='Turn off warnings')
    parser.add_argument('--eig', type=str, default='val', choices=['val','vec'], help='Choose between eigenvalues and eigenvectors')
    parser.add_argument('-p', '--power', action='store_true', help='Check the power methods using complex points')
    parser.add_argument('-c', '--cheb', action='store_true', help='Check the chebyshev methods using real points')
    parser.add_argument('--coeffs', action='store_true', help='Choose random coefficients instead of roots.')


    args = parser.parse_args()

    #assert only power or cheb
    if args.power and args.cheb:
        raise ValueError("Choose either power or chebyshev basis, but not both.")

    if not (args.power or args.cheb):
        args.power = True

    if args.nowarn:
        warnings.filterwarnings('ignore')

    # print(args.eps, type(args.eps))
    if args.num_points <= 0: raise ValueError("Not enough points")
    if args.radius <= 0: raise ValueError("Max radius must be positive")

    results = run_roots_testing(args)
    # print(results)
    create_roots_graph(args, results)
