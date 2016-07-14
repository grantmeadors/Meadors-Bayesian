#!/opt/local/bin/python
# Grant David Meadors
# 02015-04-01 (JD 2457114)
# Experimenting with some Bayesian prototypes
import numpy as np
import matplotlib as matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.backends.backend_agg

print 'Hello, world. This is a Bayesian prototype'
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
    BayesArtificialPrior = np.ones(np.shape(BayesNaturalPrior))
    BayesFirstGivenSecond = BayesFactor * BayesArtificialPrior
    return BayesFirstGivenSecond

# Implement a proper Bayesian framework given the likelihood,
def BayesCalculator(firstParam, secondParam):
    # First, we need to construct the prior on some signal parameter
    # that we know operates as intended
    # e.g., DeltaF
    # Let us say that we have a uniform prior in that first parameter;
    # by second parameter, we really mean the data
    BayesPrior = np.ones(secondParam)/secondParam
    PosteriorProbability = BayesPrior
    modelP = BayesPrior
    print modelP

    #BayesLikelihoodArray = np.log(modelP)    

    return PosteriorProbability

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

def modulatedWave(Power, DeltaF, Period, t, f, fCenter, tStart, resolution):
    tNumber = t*resolution
    fNumber = f*resolution
    prefactor = 2.0/3.0
    numerator = np.sinc(fCenter*resolution + \
    DeltaF*resolution*np.sin(2*np.pi*tNumber*tStart/Period) - \
    fNumber)**2
    denominator = 1e-9 + (\
    (fCenter*resolution+\
    DeltaF*resolution*np.sin(2*np.pi*tNumber*tStart/Period) - \
    fNumber)**2
    -1)**2
    normPowerInBin = prefactor * numerator * (1/denominator)
    # This formula does not quite work, but it gives a start
    AmplitudePsi = Power * normPowerInBin
    return AmplitudePsi

positionSamples = 20.0
timeSamples = 1.0
resolution = 10.0
positionVector = np.arange(positionSamples)/resolution
timeVector = np.arange(timeSamples)/resolution
waveObsMatrixNoise =-0.2*np.log(np.random.rand(timeSamples, positionSamples))

#waveObsMatrix = [[travellingWave(0.5, 1.0, 1.0, t, x) \
#for x in positionVector] for t in timeVector]
waveObsMatrix = [[modulatedWave(5.0, 0.3, 0.5, t, x, 1.0, 0.1, 10) \
for x in positionVector] for t in timeVector]
waveObsArray = np.asarray(waveObsMatrix)
waveObsTotal = waveObsMatrixNoise+waveObsArray

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Frequency F')
ax.set_ylabel('Time T')
ax.set_title('Bayesian non-normalized probability wave')
#extensions = [positionVector[0], positionVector[-1], timeVector[0], timeVector[-1]]
#paramSpacePixelMap = ax.imshow(waveObsTotal, origin='lower',\
#interpolation = 'nearest', extent=extensions, cmap='jet')

print positionVector/resolution
#print waveObsTotal[0]
plt.plot(positionVector/resolution, waveObsTotal[0])
plt.savefig('TravellingBayesianWave.png')
plt.close()
plt.clf()



xSamplePoint = 1.00
BayesVsTime = BayesCalculator(waveObsTotal, xSamplePoint*resolution)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Time T')
ax.set_ylabel('Posterior probability')
ax.set_title('Bayesian parameter estimationx')
plt.plot(np.arange(xSamplePoint*resolution), BayesVsTime)
plt.savefig('BayesianEstimation.png')
plt.close()
plt.clf()

