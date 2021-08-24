
import cython

cdef extern from "complex.h":
    double complex I
    double complex cexp(double complex)
    double complex clog(double complex)
    double complex cpow(double complex, double complex)
    double cabs(double complex)
    double carg(double complex)
    double cimag(double complex)
    double creal(double complex)
    
from scipy.special.cython_special cimport gammaln, gammasgn

# Already very promising. Testing with random values
# highlights that the closer to an integer c - b is,
# the worst the relative error gets.

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
@cython.cdivision(True)
cpdef double complex cff_hyp2f1a1(double b, double c, double complex z, int maxiter=1000):

    cdef double EPS = 2.0**-53

    cdef double complex C0 = 0.0j, C1 = 1.0 + 0.0j

    cdef double complex an, tn, rhon, rhonm1
    cdef int n, k

    cdef double complex res, lres, llres

    if z == C0:
        return C1

    if z == C1:
        return (c - 1.0) / (c - b - 1.0) + 0j
    
    if c == 1.0:
        return cpow(1.0 - z, -b)

    if cimag(z) == 0.0 and creal(z) > 1.0:
        # If we are directly on the branch cut, don't bother
        # letting the algo hit maxiter first and switch 
        # to the analytically continued expression right away.
        return continued_hyp2f1a1(b, c, z, maxiter=-maxiter)
    
    tn = C1
    rhon = C0
    
    n = 1

    res, lres, llres = tn, C0, C0

    while True:
        n += 1
        
        if n%2: # Odd n
            k = (n - 1)/2
            an = k * (c - 1 - b + k) * z / (c + n - 3) / (c + n - 2)

        else: # Even n
            k = (n - 2)/2
            an = (c - 1 + k) * (b + k) * z / (c + n - 3) / (c + n - 2)

        rhon = an*(C1 + rhon)/(C1 - an*(C1 + rhon))
        tn = rhon * tn
        
        res, lres, llres = res + tn, res, lres

        if (abs(tn) < EPS * abs(res)) or ((maxiter <= 0) and (n >= -maxiter)):
            return res
        elif (maxiter > 0) and (n >= maxiter):
            # If we hit maxiter and maxiter is positive, then switch to
            # the analytically continued expression with z->1-z
            return continued_hyp2f1a1(b, c, z, maxiter=-maxiter)


@cython.cdivision(True)
cpdef double complex continued_hyp2f1a1(double b, double c, double complex z, int maxiter=1000):
    
    cdef double complex C1 = 1.0 + 0.0j

    cdef double complex logACF, logf1, logf2, logf
    cdef double s1, s2

    logACF = clog(cff_hyp2f1a1(b, 2.0 + b - c, C1 - z, maxiter=maxiter))
    
    logf1 = logACF + gammaln(c - b - 1.0) - gammaln(c - 1.0) - gammaln(c - b)
    s1 = gammasgn(c - b - 1.0) * gammasgn(c - 1.0) * gammasgn(c - b)
    
    logf2 = gammaln(1.0 + b - c) - gammaln(b) + (c - b - 1.0)*clog(C1 - z) + (1.0 - c)*clog(z)
    s2 = gammasgn(1.0 + b - c) * gammasgn(b)

    if creal(logf1) >= creal(logf2):
        logf = logf1 + clog(s1 + s2*cexp(logf2 - logf1))
    else:
        logf = logf2 + clog(s1*cexp(logf1 - logf2) + s2)

    return gammasgn(c)*cexp(gammaln(c) + logf)