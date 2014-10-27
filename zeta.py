################################################################################
#
# Author: Bastian Knippschild (B.Knippschild@gmx.de) 
# Date:   October 2014 
#
# Copyright (C) 2014 Bastian Knippschild
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
#
# Function: Computation of Luescher's Zeta Function. This program is based on 
#           arXiv:1107.5023v2, e.g. equation (5). The parameter \Lambda is set 
#           to 1 in this implementation. 
#
# For informations on input parameters see the description of the function.
#
################################################################################
#
# Performed tests:
# 1.) Against Mathematica code provided by Liuming Liu w. and w.o. tbc in cms
#     and l=0, m=0 
# 2.) Against data from arXiv:1107.5023v2 w. and w.o. moving frames and l=0, m=0
# 3.) Against data from arXiv:1011.5288 w. and w.0. moving frames and linear 
#     combinations of l=2, m=-2,0,2. 
#
#  See the test function at the very end of this file for more information!
#
################################################################################

import math
import cmath
import numpy as np
import scipy.special
import scipy.integrate

################################################################################
#
#                            Luescher's Zeta Function
#
# This is the ONLY function which should and needs to be called from outside.
#
# input: q2       : (IMPORTANT:) SQUARED scattering momentum fraction, ONLY 
#                   MANDATORY INPUT PARAMETER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#        gamma    : Lorentz factor for moving frames, see e.g. arXiv:1011.5288
#        l        : orbital quantum number
#        m        : magnetic quantum number
#        d        : total three momentum of the system. (TBC: d can be used as 
#                   a twist angle as well. The correspondance is:
#                          d = -theta/pi     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#        precision: precision of the calculation
#        verbose  : 0, no output on screen; 1, detailed output with convergence
#                   informations 
#
# return: The value of Luescher's Zeta function as a COMPLEX number.
#
# minor details: The three terms A, B, and C correspond to the three terms of
#                equation (5) in arXiv:1107.5023v2.
#
################################################################################
def Z(q2, gamma = 1.0, l = 0, m = 0, d = np.array([0., 0., 0.]), \
      precision = 10e-6, verbose = 0):
  # some small checks
  if gamma < 1.0:
    print 'Gamma must be larger or equal to 1.0'
    exit(0)
  # the computation
  res = A(q2, gamma, l, m, d, precision, verbose) + \
        B(q2, gamma, l, precision, verbose) + \
        C(q2, gamma, l, m, d, precision, verbose)
  if verbose:
    print 'Luescher Zeta function:', res
  return res
################################################################################
################################################################################


################################################################################
#
#                            IMPLEMENTATION
#
################################################################################

# Transforms an array of 3d vectors from cartesian to spherical coordinates
################################################################################
def appendSpherical_np(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) 
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

# Gives an array of 3d vectors for summation
################################################################################
def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

# Computes the vector r for the sum in term A and returns it in spherical 
# coordinates 
################################################################################
def compute_r_in_spherical_coordinates(a, d, gamma):
  out = np.zeros(a.shape)
  if (np.linalg.norm(d) == 0.0):
    for r, i in zip(a, range(0,a.shape[0])):
      out[i,:] = r/gamma
  # splitting every vector in a in parallel and orthogonal part w.r.t. d
  else:
    for r, i in zip(a, range(0,a.shape[0])):
      r_p = np.dot(r, d)/np.dot(d,d)*d 
      r_o = r-r_p
      out[i,:] = (r_p-0.5*d)/gamma + r_o
  return appendSpherical_np(out)

# Computes a part of the sum in term A
################################################################################
def compute_summands_A(a_sph, q, l, m): 
  result = 0.0 
  for r in a_sph:
    result += (np.exp(-(r[0]**2.-q)) * r[0]**l) / (r[0]**2-q) * \
              scipy.special.sph_harm(m, l, r[2], r[1]) 
  return result


# creates the momentum array used for the sums
################################################################################
def create_momentum_array(p):
  i = int(math.sqrt(p)+1)
  n = [j for j in xrange(-i,i+1)]
  r = cartesian((n, n, n))
  out = []
  for rr in r:
    if (np.dot(rr, rr) == p):
      out.append(np.ndarray.tolist(rr))
  out = np.asarray(out, dtype=float)
  p += 1
  # these momentum suqares do not exist
  exclude = [7, 15, 23, 28, 31, 39, 47, 55, 60, 63, 71, 79, 92, 112, 124, 156, \
             188, 220, 240, 252, 284, 316, 368, 448, 496, 624, 752, 880, 960]
  if p in exclude:
    p += 1
  if p > 1007:
    print 'cannot converge, see zeta.py - create_momentum_array'
    exit(0)
  return out, p

# Computation of term A
################################################################################
def A(q, gamma, l, m, d, precision, verbose):
  i = 0
  r, i = create_momentum_array(i)
  r_sph = compute_r_in_spherical_coordinates(r, d, gamma)
  result = compute_summands_A(r_sph, q, l, m)
  if verbose:
    print 'convergence in term A:'
    print '\t', i-1, result
  # computing new sums until precision is reached
  eps = 1
  while (eps > precision):
    r, i = create_momentum_array(i)
    r_sph = compute_r_in_spherical_coordinates(r, d, gamma)
    result_h = compute_summands_A(r_sph, q, l, m)
    eps = abs(result_h/result)
    result += result_h
    if verbose:
      print '\t', i-1, result, eps
  if verbose:
    print 'Term A:', result
  return result

# Computation of term B
################################################################################
def B(q, gamma, l, precision, verbose):
  if l is not 0:
    return 0.0 
  else:
    a = 2.*scipy.special.sph_harm(0, 0, 0.0, 0.0)*gamma*math.pow(math.pi, 3./2.)
    # The integral gives [2*(exp(q)*DawsonF(sqrt(q))/sqrt(q)] for Lambda = 1
    # The Dawson function is only available in scipy 0.13 or so, so it is
    # replaced by a representation with the Gaussian error function.
    dawson = -1.j*np.sqrt(math.pi) * np.exp(-q) * \
             scipy.special.erf(1.j*cmath.sqrt(q)) / 2.
    b = q * 2.*np.exp(q)*dawson/cmath.sqrt(q)
    c = math.exp(q)
    if verbose:
      print 'Term B:', a*(b-c)
    return a*(b-c)


# Computes the term gamma*w and returns the result in spherical coordinates
################################################################################
def compute_gamma_w_in_spherical_coordinates(a, d, gamma):
  out = np.zeros(a.shape)
  if (np.linalg.norm(d) == 0.0):
    for r, i in zip(a, range(0,a.shape[0])):
      out[i,:] = r*gamma
  # splitting every vector in a in parallel and orthogonal part w.r.t. d
  else:
    for r, i in zip(a, range(0,a.shape[0])):
      r_p = np.dot(r, d)/np.dot(d,d)*d 
      r_o = r-r_p
      out[i,:] = r_p*gamma + r_o
  return appendSpherical_np(out)

# Just the integrand of term C
################################################################################
integrand = lambda t, q, l, w: ((math.pi/t)**(3./2.+l) ) * \
                               np.exp(q*t-(math.pi)**2.*w*w/t)

# Computes a part of the sum in term C
################################################################################
def compute_summands_C(w_sph, w, q, gamma, l, m, d, precision):
  part1 = gamma * (np.absolute(w_sph[:,0])**l) * \
          np.exp((-1.j)*math.pi*np.dot(w, d)) * \
          scipy.special.sph_harm(m, l, w_sph[:,2], w_sph[:,1])
  # Factor two: The integral 
  part2 = []
  for ww in w_sph:
    # the precision in this integral might be crucial at some point but it is
    # very high right now with a standard of 1e-12. It should be enough
    # for all comoputations. In doubt, please change it.
    part2.append((scipy.integrate.quadrature(integrand, 0., 1., \
                  args=(q, l, ww[0]), tol = precision*1e-6, maxiter=1000))[0])
  part2 = np.asarray(part2, dtype=float)
  # return the result
  return np.dot(part1, part2)

# Computation of term C
################################################################################
def C(q, gamma, l, m, d, precision, verbose):
  i = 1
  w, i = create_momentum_array(i)
  w_sph = compute_gamma_w_in_spherical_coordinates(w, d, gamma)
  result = compute_summands_C(w_sph, w, q, gamma, l, m, d, precision)
  if verbose:
    print 'convergence in term C:'
    print '\t', i-1, result
  # computing new sums until precision is reached
  eps = 1
  while (eps > precision):
    w, i = create_momentum_array(i)
    w_sph = compute_gamma_w_in_spherical_coordinates(w, d, gamma)
    result_h = compute_summands_C(w_sph, w, q, gamma, l, m, d, precision)
    eps = abs(result_h/result)
    result += result_h
    if verbose:
      print '\t', i-1, result, eps
  if verbose:
    print 'Term C:', result
  return result


def test(): 
  # cms ##########################
  print '\nTest in cms:'
  Pcm = np.array([0., 0., 0.])
  q = 0.1207*24/(2.*math.pi)
  gamma = 1.0
  zeta = Z(q*q, gamma, d = Pcm).real
  print 'q, gamma:', q, gamma
  delta = np.arctan(math.pi**(3./2.)*q/zeta)*180./math.pi
  if delta < 0:
    delta = 180+delta
  print 'delta:', delta, 'delta should be: 137'
  
  # mv1 ##########################
  print '\nTest in mv1:'
  Pcm = np.array([0., 0., 1.])
  L = 32
  q = 0.161*L/(2.*math.pi)
  E = 0.440
  Ecm = 0.396
  gamma = E/Ecm
  Z00 = Z(q*q, gamma, d = Pcm).real
  Z20 = Z(q*q, gamma, d = Pcm, l = 2).real
  print 'q, gamma:', q, gamma
  delta = np.arctan(gamma*math.pi**(3./2.) * q / \
          (Z00 + (2./(q*q*math.sqrt(5)))*Z20))*180./math.pi
  if delta < 0:
    delta = 180+delta
  print 'delta:', delta, 'delta should be: 116'
  
  
  # mv2 ##########################
  print '\nTest in mv2:'
  Pcm = np.array([1., 1., 0.])
  L = 32
  q = 0.167*L/(2.*math.pi)
  E = 0.490
  Ecm = 0.407
  gamma = E/Ecm
  Z00 = Z(q*q, gamma, d = Pcm).real
  Z20 = Z(q*q, gamma, d = Pcm, l = 2).real
  Z22  = Z(q*q, gamma, d = Pcm, l = 2, m = 2).imag
  Z2_2 = Z(q*q, gamma, d = Pcm, l = 2, m = -2).imag
  print 'q, gamma:', q, gamma
  delta = np.arctan(gamma*math.pi**(3./2.) * q / \
          (Z00 - (1./(q*q*math.sqrt(5)))*Z20 \
          + ((math.sqrt(3./10.)/(q*q))*(Z22-Z2_2))))*180./math.pi
  if delta < 0:
    delta = 180+delta
  print 'delta:', delta, 'delta should be: 128'

