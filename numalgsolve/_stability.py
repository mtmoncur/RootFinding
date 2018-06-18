import numpy as np
from numalgsolve.polynomial import MultiCheb, MultiPower
from numalgsolve.OneDimension import multPowerR
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


def create_roots_graph(results):
    plt.figure(figsize=(14,6))
    nrows = len(results)
    ncols = len(next(iter(results.values()))) - 1 #get sub dictionary length, but ignore the roots item
    for i,(radius, sub_results) in enumerate(results.items()):
        radius = float(radius)
        roots = sub_results['roots']
        del sub_results['roots']
        for j,(method, roots_approx) in enumerate(sub_results.items()):
            ax = plt.subplot(nrows,ncols,i*ncols+(j+1))
            plt.plot(roots.real, roots.imag, 'r+', ms = 10)
            plt.plot(roots_approx.real, roots_approx.imag, 'b*')
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


def run_roots_testing(args):

    shrink_factor = 0.7
    radius = args.radius
    num_points = args.points
    eps = args.eps
    results = {}
    for _ in range(3):
        # create random complex roots
        r = np.random.random(num_points)*radius + eps
        args = 2*np.pi*np.random.random(num_points)
        # roots = r*np.exp(args*1j)
        roots = r

        # results['roots#{:.2f}'.format(radius)] = roots
        r_trunc = '{:.2f}'.format(radius)
        results[r_trunc] = {'roots':roots}

        powerpoly = [MultiPower(polyfromroots(roots))]
        chebpoly = [MultiCheb(chebfromroots(roots))]
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

        # results[r_trunc]['mult power']  = solve(powerpoly, 'mult')
        results[r_trunc]['multR power']  = multPowerR(powerpoly[0].coeff)
        results[r_trunc]['div power']   = solve(powerpoly, 'div')
        results[r_trunc]['mult cheb']   = solve(chebpoly, 'mult')
        from numalgsolve.OneDimension import div2Cheb
        results[r_trunc]['div cheb']    = solve(chebpoly, 'div')
        results[r_trunc]['div2 cheb']    = div2Cheb(chebpoly[0].coeff)
        results[r_trunc]['numpy power'] = polyroots(powerpoly[0].coeff)
        results[r_trunc]['numpy cheb']  = chebroots(chebpoly[0].coeff)

        plt.subplot(121)
        plt.hist(np.abs(powerpoly[0](results[r_trunc]['multR power'])))
        plt.title("multR eval at roots")
        plt.subplot(122)
        plt.hist(np.abs(powerpoly[0](results[r_trunc]['div2 cheb'])))
        plt.title("div2 eval at roots")
        plt.show()


        radius *= shrink_factor
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Stability Test Options")
    parser.add_argument('-p', '--points', type=int, default=50, help='Number of complex roots, minimum of 2')
    parser.add_argument('-r', '--radius', type=float, default=1, help='The largest radius for the points')
    parser.add_argument('-e', '--eps', type=float, default=1e-8, help='Number of complex roots, minimum of 2')
    parser.add_argument('-W', '--nowarn', action='store_true', help='Turn off warnings')

    args = parser.parse_args()

    if args.nowarn:
        warnings.filterwarnings('ignore')

    print(args.eps, type(args.eps))
    if args.points <= 0: raise ValueError("Not enough points")
    if args.radius <= 0: raise ValueError("Max radius must be positive")

    results = run_roots_testing(args)
    # print(results)
    create_roots_graph(results)
