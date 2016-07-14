#!/opt/local/bin/python
# Grant David Meadors
# 02015-03-31 (JD 2457113)
# Experimenting with some Bayesian examples
import numpy as np
import matplotlib as matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.backends.backend_agg

print 'Hello, world. This is a Bayesian example'
# Bayes theorem:
# P(A|B) = ( P(A|B) / P(B) ) * P(A)

def BayesTheoremTwoParameter(obsMatrix, firstParam, secondParam):
    # Return the probability of the firstParam in the arguments,
    # given the secondParam in the arguments
    # Originally designed where firstParam can be
    # 0 for NS or 1 for BH
    # and the secondParam can be
    # 0 for C or 1 for I
    BayesNaturalPrior = np.sum(obsMatrix/np.sum(obsMatrix), tuple(x for x in (0,1) if x!=0) )[firstParam]
    ProbFirstGivenSecond = \
    ( ( (obsMatrix[firstParam,secondParam]/np.sum(obsMatrix)) / (np.sum(obsMatrix[firstParam,:])/np.sum(obsMatrix)) ) )
    ProbSecond = ( \
    np.sum(obsMatrix/np.sum(obsMatrix), tuple(x for x in (0,1) if x!=1) )[secondParam])
    BayesFactor = ProbFirstGivenSecond / ProbSecond
    
    #print 'Things of interest'
    #print obsMatrix[firstParam,secondParam] / np.sum(obsMatrix[firstParam,:])
    #print obsMatrix[firstParam,secondParam]
    #print np.sum(obsMatrix[firstParam,:])

    #print np.sum(obsMatrix/np.sum(obsMatrix), 0)
    #print 'ProbSecond'
    #print ProbSecond
    #print 'Bayes Natural Prior'
    #print BayesNaturalPrior
    BayesArtificialPrior = np.ones(np.shape(BayesNaturalPrior))
    #BayesFirstGivenSecond = BayesFactor * BayesNaturalPrior
    BayesFirstGivenSecond = BayesFactor * BayesArtificialPrior
    #print BayesFactor
    print BayesFirstGivenSecond

    # Diagnostic stages, comment as needed
    #print 'Input matrix'
    #print obsMatrix
    return BayesFirstGivenSecond

# Let us say that we have a set of sensors recording readings
# at a set of times and place:
# t0x0 t0x1 t0x2 ...
# t1x0 t1x1 t1x2 ...
# t2x0 t2x1 t2x2 ...
# Imagine a pressure wave passing by. Suppose we can only observe
# at one point, e.g., x2, and interpret pressure somewhat quantum
# mechanically, the probability of an object being there.
# When we want to know P(T1 | X2), the interpretation probably
# is more that we want to know what is the probability that the
# peak is there. But it is hard to say. So let us construct an
# example.

def travellingWave(A, v, wavelength, t, x):
    AmplitudePsi = A * np.sin((2*np.pi/wavelength)*(x - v*t))
    return AmplitudePsi
positionSamples = 20.0
timeSamples = 30.0
resolution = 10.0
positionVector = np.arange(positionSamples)/resolution
timeVector = np.arange(timeSamples)/resolution
#print positionVector
#print timeVector
waveObsMatrixNoise = 1.0+np.random.rand(timeSamples, positionSamples)
#print waveObsMatrixNoise

waveObsMatrix = [[travellingWave(0.5, 1.0, 1.0, t, x) \
for x in positionVector] for t in timeVector]
waveObsArray = np.asarray(waveObsMatrix)
#print waveObsArray
#print np.shape(waveObsMatrixNoise+waveObsArray)
waveObsTotal = waveObsMatrixNoise+waveObsArray
#print waveObsTotal

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Position X')
ax.set_ylabel('Time T')
ax.set_title('Bayesian non-normalized probability wave')
extensions = [positionVector[0], positionVector[-1], timeVector[0], timeVector[-1]]
paramSpacePixelMap = ax.imshow(waveObsTotal, origin='lower',\
interpolation = 'nearest', extent=extensions, cmap='jet')
plt.savefig('TravellingBayesianWave.png')
plt.close()
plt.clf()

# It is quite straightforward, although not completely obvious what it means, 
# now to try to find the posterior probability of time = 0.5, 1.5, and 2.5,
# given x=0.75 and 1.25, for instance
#print 'Bayesian posteriors on time = 0.5, 1.5, 2.5, given x = 0.75'
#print BayesTheoremTwoParameter(waveObsTotal, 0.5, 0.75)
#print BayesTheoremTwoParameter(waveObsTotal, 1.5, 0.75)
#print BayesTheoremTwoParameter(waveObsTotal, 2.5, 0.75)
#print 'Bayesian posteriors on time = 0.5, 1.5, 2.5, given x = 1.25'
#print BayesTheoremTwoParameter(waveObsTotal, 0.5, 1.25)
#print BayesTheoremTwoParameter(waveObsTotal, 1.5, 1.25)
#print BayesTheoremTwoParameter(waveObsTotal, 2.5, 1.25)

# So that is not very interesting, since each of them is about the same,
# 1/30, with some fluctuations do to noise. But what is we put a more
# interesting phenomenological model in, e.g., a peak?

def GaussianPeak(A, meanT, meanX, t, x):
    #AmplitudePsi = A * np.exp((-(x-meanX)**2 -(t-meanT)**2)/2)
    # The Gaussian above has this unfortunate issue, where the 
    # value of the Gaussian at any given point scales with the
    # conditional probability. So, let us try different
    AmplitudePsi = 5*A * np.exp((-((x-meanX)/0.3)**2 -((t-meanT)/0.15)**2)/2)+\
    3*A * np.exp((-((x-1.0)/0.1)**4-((t-1.0)/0.2)**4)/2)
    return AmplitudePsi

GaussObsMatrix = [[GaussianPeak(5.5, 1.55, 0.95, t, x) \
for x in positionVector] for t in timeVector]
GaussObsArray = np.asarray(GaussObsMatrix)
GaussObsTotal = waveObsMatrixNoise+GaussObsArray

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Position X')
ax.set_ylabel('Time T')
ax.set_title('Bayesian non-normalized probability surge')
extensions = [positionVector[0], positionVector[-1], timeVector[0], timeVector[-1]]
paramSpacePixelMap = ax.imshow(GaussObsTotal, origin='lower',\
interpolation = 'nearest', extent=extensions, cmap='jet')
plt.savefig('BayesianGaussian.png')
plt.close()
plt.clf()


# It is quite straightforward, although not completely obvious what it means, 
# now to try to find the posterior probability of time = 0.5, 1.5, and 2.5,
# given x=0.75 and 1.25, for instance
#print 'GAUSSIAN SURGE SECTION'
#print 'Bayesian posteriors on time = 0.5, 1.5, 2.5, given x = 0.75'
#print BayesTheoremTwoParameter(GaussObsTotal, 0.5, 0.75)
#print BayesTheoremTwoParameter(GaussObsTotal, 1.5, 0.75)
#print BayesTheoremTwoParameter(GaussObsTotal, 2.5, 0.75)
#print 'Bayesian posteriors on time = 0.5, 1.5, 2.5, given x = 1.00'
#print BayesTheoremTwoParameter(GaussObsTotal, 0.5, 1.00)
#print BayesTheoremTwoParameter(GaussObsTotal, 1.5, 1.00)
#print BayesTheoremTwoParameter(GaussObsTotal, 2.5, 1.00)
#print 'Bayesian posteriors on time = 0.5, 1.5, 2.5, given x = 1.25'
#print BayesTheoremTwoParameter(GaussObsTotal, 0.5, 1.25)
#print BayesTheoremTwoParameter(GaussObsTotal, 1.5, 1.25)
#print BayesTheoremTwoParameter(GaussObsTotal, 2.5, 1.25)
#print 'This Gaussian parameter estimation section does not work right,'
#print 'I think, but it is a start'
#print 'Problem SOLVED: one needs to multiple the parameter arguments'
#print 'by the sampling resolution, which should've been obvious'

# Try some serious Bayesian parameter estimation
#
xSamplePoint = 1.00
BayesVsTime = [BayesTheoremTwoParameter(GaussObsTotal, resolution*time, xSamplePoint*resolution) for time in timeVector]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Time T')
ax.set_ylabel('Posterior probability')
ax.set_title('Bayesian parameter estimation at sample point x')
plt.plot(timeVector, BayesVsTime)
plt.savefig('BayesianEstimation.png')
plt.close()
plt.clf()

