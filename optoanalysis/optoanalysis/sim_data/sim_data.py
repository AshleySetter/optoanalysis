from optoanalysis import DataObject
from optoanalysis.sde_solver import sde_solver
import numpy as _np
from multiprocessing import Pool as _Pool

class SimData(DataObject):
    """
    
    """
    def __init__(self, TimeTuple, SampleFreq, TrapFreqArray, Gamma0, mass, ConvFactor, NoiseStdDev, T0=300.0, etaArray=None, dt=1e-10, seed=None):
        """
        
        """
        self.q0 = 0.0
        self.v0 = 0.0
        self.TimeTuple = (TimeTuple[0], TimeTuple[1])
        self.SampleFreq = SampleFreq
        self.TrapFreqArray = _np.array(TrapFreqArray)
        self.Gamma0 = Gamma0
        self.mass = mass
        self.ConvFactor = ConvFactor
        self.NoiseStdDev = NoiseStdDev
        self.T0 = T0
        if etaArray == None:
            self.etaArray = _np.zeros_like(TrapFreqArray)
        self.dt = dt
        self.seed = seed        
        self.generate_simulated_data()
        dtSample = 1/SampleFreq
        self.DownSampleAmount = dtSample/dt
        return None

    def generate_simulated_data(self):
        self.sde_solvers = []
        if self.seed != None:
            _np.random.seed(self.seed)
        for i, freq in enumerate(self.TrapFreqArray):
            TrapOmega = freq*2*_np.pi
            solver = sde_solver(TrapOmega, self.Gamma0, self.mass, eta=self.etaArray[i], T0=self.T0, q0=self.q0, v0=self.v0, TimeTuple=self.TimeTuple, dt=self.dt)
            self.sde_solvers.append(solver)
        #workerPool = _Pool()
        #workerPool.map(run_solve, self.sde_solvers)
        for solver in self.sde_solvers:
            print('solving...')
            solver.solve()
        return None

def run_solve(sde_solver):
    print('solving...')
    sde_solver.q, sde_solver.v = sde_solver.solve()
    return None
