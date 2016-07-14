#!/opt/local/bin/python
# Grant David Meadors
# 02015-03-31 (JD 2457113)
# Experimenting with some preliminary Bayesian methods in sandbox
import numpy as np

print 'Hello, world. This is a Bayesian sandbox'
# Bayes theorem:
# P(A|B) = ( P(A|B) / P(B) ) * P(A)

# Suppose we have some extremely simple problem.
# Suppose there are multiple kinds of signal,
# and we want to know what will be detected first.
# Not simple enough.
# Suppose we have a set of sources with some discrete properties
# For instace, type of star, and whether or not it has a companion.
# So, is it a neutron star (NS) or black hole (BH),
# and does it have a companion (C) or it is isolated (I)?
# There are a total of four cases, so let's say the total number of
# observed cases of each is (to invent numbers)
# NS-C: 4
# NS-I: 8
# BH-C: 3
# BH-I: 1
# Then how do we go about this problem?
# Suppose we want to figure out the probability of being an NS or BH.
# Let us construct a matrix:

obsMatrix = np.array([[4., 8.],[3., 1.]])
print 'Matrix of observations:'
print 'NS-C, NS-I;'
print 'BH-C, BH-I'
print obsMatrix

# Neutron stars are in the first row, black holes in the second.
# Stars with companions are in the first column, isolated stars in the second.
# So we have a matrix,
# P_{ab}
# Where to find out P(A) for A, a specific value of a, 
# we compute P(A) = \Sigma_b P_{ab} / (\Sigma_{ab} P_{ab}),
# that is, marginalize over everything except a in the numerator.

# The denominator is so ubiquitous we may want to write a matrix 
# just normalized to it

print 'Matrix of probabilities:'
probMatrix = obsMatrix / np.sum(obsMatrix)
print probMatrix

print 'Probability of neutron stars and black holes, computed from first matrix:'
# First sum over columns, because we are marginalizing over all except rows
# Note python counts from 0, so row are 0, and columns, being the second axis, are 1
obsMatrixRows = np.sum(obsMatrix,1) / np.sum(obsMatrix)
print obsMatrixRows
print 'Probability of neutron stars, computed from first matrix:'
obsMatrixNS = obsMatrixRows[0]
print obsMatrixNS
print 'Probability of neutron stars and black holes, computed from second matrix:'
# Do ditto
probMatrixRows = np.sum(probMatrix,1)
print probMatrixRows
print 'Probability of neutron stars, computed from second matrix:'
probMatrixNS = probMatrixRows[0]
print probMatrixNS

# That is sufficient demonstration that the probability matrix is fine for our purposes.
# Assign the corresponding factors,
probNS = np.sum(probMatrix,1)[0]
probBH = np.sum(probMatrix,1)[1]
probC = np.sum(probMatrix,0)[0]
probI = np.sum(probMatrix,0)[1]
# Print the overall probabilities
print 'Relative probabilities of being neutrons stars, black holes, companioned or isolated:'
print probNS
print probBH
print probC
print probI

print 'Highlight: probability that a star is isolated, P(I):'
print probI

# All this is relatively obvious from the matrix, by inspection

# Excellent. Now we need the cognitive leap: conditional probability
# So, what is the probability that something is a neutron star, given that is
# an isolated star, for instance? That is not so obvious.
# Bayes Theorem would say P(NS|I) = (P(I|NS)/P(I)) * P(NS)
# We already saw how to compute marginalize P(I) and P(NS); 
# the P(I|NS) and P(NS|I) are really not any harder
# The given implies a restriction, and then we just use the overall probability
# in that domain as the new numerator and the probability of that particular
# case as the numerator. In general the numerator might be another sum or integral,
# but our case is simple, so it is just a point.
probIgivenNS = probMatrix[0,1]/np.sum(probMatrix[0,:])
print 'Probability a star is isolated, given it is a neutron star, P(I|NS)'
print probIgivenNS
# Likewise we can compute the whole set
probCgivenNS = probMatrix[0,0]/np.sum(probMatrix[0,:])
probIgivenBH = probMatrix[1,1]/np.sum(probMatrix[1,:])
probCgivenBH = probMatrix[1,0]/np.sum(probMatrix[1,:])
# We can also, since we have access to the full data set, invert the problem and
# compute the other half of teh equal, which we will use to check the theorem:
probNSgivenC = probMatrix[0,0]/np.sum(probMatrix[:,0])
probNSgivenI = probMatrix[0,1]/np.sum(probMatrix[:,1])
probBHgivenC = probMatrix[1,0]/np.sum(probMatrix[:,0])
probBHgivenI = probMatrix[1,1]/np.sum(probMatrix[:,1])

print 'Probability a star is a neutron star, given it is isolated, computing directly P(NS|I):'
print probNSgivenI

# So, at last, the theorem of Bayes tells us that we should get the same answer if
# we proceed in the other direction:
print 'Probability a star is a neutron star, given it is isolated, computing Bayes...'
print 'Bayes factor, P(I|NS) / P(I):'
BayesFactorNSgivenI = (probIgivenNS / probI)
print BayesFactorNSgivenI
print 'Prior probability, P(NS):'
BayesPriorNS = probNS
print BayesPriorNS
print 'Bayes theorem result, P(NS|I):' 
BayesTheoremNSgivenI = BayesFactorNSgivenI * BayesPriorNS
print BayesTheoremNSgivenI

# Therefore, we have demonstrated that Bayes theorem works.
# We have nicely divided up the problem so that the steps are clear,
# but how does this all work in practice? If we expanded everything back
# to the matrix of observations, we would get...
# Notice that a few terms do seem to cancel out
obsMatrixBayesTheoremNSgivenI = \
(( (obsMatrix[0,1]/np.sum(obsMatrix)) / np.sum(obsMatrix[0,:]/np.sum(obsMatrix)) ) / \
 (np.sum(obsMatrix,0)/np.sum(obsMatrix))[1] ) \
* \
 (np.sum(obsMatrix,1)/np.sum(obsMatrix))[0]
print 'Bayes theorem result from observation matrix without intermediates,'
print obsMatrixBayesTheoremNSgivenI
# As we see, this can be, at least in this simple two-parameter model,
# greatly expedited the cancelling out the overall normalization, per
print 'Bayes theorem result from observation matrix, simplified calculation:'
obsMatrixBayesTheoremNSgivenIsimple = \
( obsMatrix[0,1] / np.sum(obsMatrix[0,:]) / \
 np.sum(obsMatrix,0)[1]) \
* \
  np.sum(obsMatrix,1)[0]
print obsMatrixBayesTheoremNSgivenIsimple

# From my understanding, if we had additional dimensions, we could
# survive with these cancellations, but the sums over axes would
# need to be edited; the sums would have to be ALL axes except the ones
# being recovered.
# The more general way to do the marginalizations would be...
obsMatrixBayesTheoremNSgivenIsimpleGeneral = \
( obsMatrix[0,1] / np.sum(obsMatrix[0,:]) / \
 np.sum(obsMatrix,tuple(x for x in (0,1) if x != 1))[1]) \
* \
  np.sum(obsMatrix,tuple(x for x in (0,1) if x !=0))[0]
# IMPORTANT: do not read in the !=1 and [1] or !=0 and [0]
# being on the same line to interpret those as related.
# It is just coincidence that I chose two things
# on the "diagonal" (not of the matrix, of the the parameter space definitions),
# being interested in isolated stars, which are the second item defined
# by the second parameter (companion vs isolated)
# and in neutron stars, which are the first item defined by the first
# paramter (neutron star vs black hole)
print 'Bayes, simplified then generalized:'
print obsMatrixBayesTheoremNSgivenIsimpleGeneral

# So, for one hundred percent numerical completeness,
# If we throw the observation matrix normalizations back in,
# together with these generalized tuples, we can get a function
def BayesTheoremTwoParameter(obsMatrix, firstParam, secondParam):
    # Return the probability of the firstParam in the arguments,
    # given the secondParam in the arguments
    # Originally designed where firstParam can be
    # 0 for NS or 1 for BH
    # and the secondParam can be
    # 0 for C or 1 for I
    BayesFirstGivenSecond = \
    ( ( (obsMatrix[firstParam,secondParam]/np.sum(obsMatrix)) / (np.sum(obsMatrix[firstParam,:])/np.sum(obsMatrix)) ) / \
    np.sum(obsMatrix/np.sum(obsMatrix), tuple(x for x in (0,1) if x!=1) )[secondParam]) \
    * \
    np.sum(obsMatrix/np.sum(obsMatrix), tuple(x for x in (0,1) if x!=0) )[firstParam]
    return BayesFirstGivenSecond
print 'Fully general Bayes theorem specification of NS given I:'
print BayesTheoremTwoParameter(obsMatrix, 0,1)

# Let us conclude by showing that the function works in general,
print 'Ditto, NS given C:'
print BayesTheoremTwoParameter(obsMatrix, 0,0)
print 'Ditto, BH given I:'
print BayesTheoremTwoParameter(obsMatrix, 1,1)
print 'Ditto, BH given C:'
print BayesTheoremTwoParameter(obsMatrix, 1,0)

# To be one-hundred percent sure, let's swap the axes on the array and
# verify that we get the reverse conditionals as expected
# TURNS OUT we have to swap the firstParam and secondParam arguments then,
print 'Fully general Bayes theorem: C given NS:'
print BayesTheoremTwoParameter(obsMatrix.swapaxes(0,1), 0, 0)
print 'Fully general Bayes theorem: I given NS:'
print BayesTheoremTwoParameter(obsMatrix.swapaxes(0,1), 1, 0)
print 'Fully general Bayes theorem: C given BH:'
print BayesTheoremTwoParameter(obsMatrix.swapaxes(0,1), 0, 1)
print 'Fully general Bayes theorem: I given BH:'
print BayesTheoremTwoParameter(obsMatrix.swapaxes(0,1), 1, 1)

# So, all the examples of Bayes theorem are established, and math works
