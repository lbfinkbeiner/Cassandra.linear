from scipy.interpolate import UnivariateSpline
import scipy.fft as fft

def get_nowiggle_pk(kfinal, karr, Parr, omega_m, range_imin=np.array([80,150]),
    range_imax=np.array([200,300]), threshold=0.04, offset=-25):

    logParr = np.log(karr*Parr)
    xiarr = fft.dst(logParr, type=2)
    xi_even = xiarr[::2]
    xi_odd = xiarr[1::2]
    xi_even_spline = UnivariateSpline(np.arange(2**15),xiarr[::2],k=3,s=0)
    xi_odd_spline = UnivariateSpline(np.arange(2**15),xiarr[1::2],k=3,s=0)
    
    range_imin = (range_imin*(0.1376591/omega_m)**(1./4)).astype(int)
    range_imax = (range_imax*(0.1376591/omega_m)**(1./4)).astype(int)
    spline_domain_imin = np.arange(range_imin[0],range_imin[1])
    spline_domain_imax = np.arange(range_imax[0],range_imax[1]) 
    spline_imin = xi_even_spline.derivative(n=2)(spline_domain_imin))
    spline_imax = xi_even_spline.derivative(n=2)(spline_domain_imax) 

    imin_start = np.argmin(spline_imin) + range_imin[0] + offset
    imax_start = np.where(spline_imax < threshold)[0][0] + range_imax[0]
    
    def remove_bump(imin, imax):
        r = np.delete(np.arange(2**15), np.arange(imin,imax))
        xi_even_nobumb = np.delete(xi_even, np.arange(imin,imax))
        xi_odd_nobumb = np.delete(xi_odd, np.arange(imin,imax))
        xi_even_nobumb_spline = UnivariateSpline(r, (r+1)**2*xi_even_nobumb,
                                                 k=3, s=0)
        xi_odd_nobumb_spline = UnivariateSpline(r, (r+1)**2*xi_odd_nobumb,
                                                k=3, s=0)

        xi_nobumb = np.zeros(2**16)
        xi_nobumb[::2] = xi_even_nobumb_spline(np.arange(2**15)) / \
            np.arange(1,2**15+1)**2
        xi_nobumb[1::2] = xi_odd_nobumb_spline(np.arange(2**15)) / \
            np.arange(1,2**15+1)**2

        logkpk_nowiggle = fft.idst(xi_nobumb, type=2)
        return UnivariateSpline(karr, np.exp(logkpk_nowiggle)/karr, k=3, s=0)
    
    P = UnivariateSpline(karr, Parr, k=3, s=0)
    Pnw = remove_bump(imin_start, imax_start)
        
    return P(kfinal), Pnw(kfinal)
