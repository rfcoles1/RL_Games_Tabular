import numpy as np
import gym

LatentHeat_Fusion = 334 #kJ/kg
LatentHeat_Vapor = 2264.705 #kJ/kg
MeltingPoint = 0 #degC
BoilingPoint = 100.0 #degC
HeatCap_Ice = 2.108 #kJ/kg/C
HeatCap_Water = 4.148 #kJ/kg/C
HeatCap_Steam = 1.996 #kJ/kg/C

class WaterEnvTab(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        
        self.RoomTemp = 21 #degC
        self.gen_critPoints()

        self.minT = -50 #degC
        self.maxT = 150 #degC
        self.minE = self.get_Energy(self.minT) #kJ
        self.maxE = self.get_Energy(self.maxT) #kJ
        
        self.statesize = 100
        
        #state table
        self.Energies = np.linspace(self.minE, self.maxE, self.statesize)
    
        self.TargetTemp = 100. #degC
        self.TargetMass = np.array([0,.5,.5]) #Sum to 1

        self.reset()
                
    def reset(self):
        self.stateID = np.argmin(abs(self.Energies))
        self.done = False
        return self.get_state()

    def get_state(self):
        E = self.Energies[self.stateID]
        T,M = self.get_true_value(E)
        print(T,M) 
        T_norm = (T - self.minT)/(self.maxT - self.minT)
        M_norm = self.encodeMass(M)/2.0

        return np.array([T_norm, M_norm])


    def encodeMass(self, m):
        m = np.array(m)
        tmp = np.where(m>0)[0]
        maxval = np.max(tmp)
        return(maxval - 1 + m[maxval])

    def decodeMass(self, M):
        massfrac = np.zeros(3)
        #the gp model is not bounded outside of observed data so we must clip
        M = np.clip(M, 0, 2)
        mInd = int(np.ceil(M))                                                                          
        if M%1 != 0:
            massfrac[mInd] = M%1
            massfrac[mInd - 1] = 1-M%1
        else:
            massfrac[mInd] = 1
        return massfrac

    # 0 - move back
    # 1 - stay
    # 2 - move forward
    def step(self, action):
        #self.stateID += int(action-1)
        if action == 0 and self.stateID > 0:
            self.stateID -= 1
        if action == 2 and self.stateID < len(self.Energies)-1:
            self.stateID += 1
        
        next_state = self.get_state()
        reward = self.get_reward()
        return next_state, reward, self.done, {}

    def get_reward(self):
        state = self.get_state()
       
        TargetT = (self.TargetTemp - self.minT)/(self.maxT - self.minT)
        T_reward = -abs(TargetT - state[0])
        
        TargetM = self.encodeMass(self.TargetMass)/2.0 
        M_reward = -abs(TargetM - state[1])
        return T_reward + M_reward





    def get_Energy(self, T): 
    #return energy for specific T, wont be necessarily correct for Boiling/Melting temperature
        if T < MeltingPoint:
            E = self.Lower_Melting_Energy + -(abs(T-MeltingPoint)*HeatCap_Ice)
        elif T < BoilingPoint:
            E = self.Upper_Melting_Energy + ((T - MeltingPoint)*HeatCap_Water)
        else:
            E = self.Upper_Boiling_Energy + (abs(T-BoilingPoint)*HeatCap_Steam)
        return E 

    def get_true_value(self, E): 
    #gives true values for temperature and mass fraction, given an energy value
        if E > self.Upper_Boiling_Energy:
            T = BoilingPoint + (1./HeatCap_Steam)*(E-self.Upper_Boiling_Energy)
            MassFractions = [0,0,1]
            return T, MassFractions
        elif E > self.Lower_Boiling_Energy:
            T = BoilingPoint
            Ediff = E - self.Lower_Boiling_Energy
            MassFractions = [0, 1. - Ediff/LatentHeat_Vapor, Ediff/LatentHeat_Vapor]
            return T, MassFractions
        elif E > self.Upper_Melting_Energy:
            T = MeltingPoint + (1./HeatCap_Water)*(E-self.Upper_Melting_Energy)
            MassFractions = [0,1,0]
            return T, MassFractions
        elif E > self.Lower_Melting_Energy:
            T = MeltingPoint
            Ediff = E - self.Lower_Melting_Energy
            MassFractions = [1. - Ediff/LatentHeat_Fusion, Ediff/LatentHeat_Fusion, 0]
            return T, MassFractions
        else:
            T = MeltingPoint + (1./HeatCap_Ice)*(E - self.Lower_Melting_Energy)
            MassFractions = [1,0,0]
            return T, MassFractions 

    def gen_critPoints(self):    
    #given a refernece temp, returns the amount of total energy for a phase transition 
        self.Upper_Melting_Energy = (MeltingPoint - self.RoomTemp) * HeatCap_Water
        self.Lower_Melting_Energy = self.Upper_Melting_Energy - LatentHeat_Fusion

        self.Lower_Boiling_Energy = (BoilingPoint - self.RoomTemp) * HeatCap_Water
        self.Upper_Boiling_Energy = self.Lower_Boiling_Energy + LatentHeat_Vapor



