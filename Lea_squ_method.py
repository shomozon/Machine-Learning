import numpy as np
import matplotlib.pyplot as plt

# -----最小二乗法の class-----

class Lea_squ_method:

    def __init__(self, N, M, x, t):
    
        self.N = N
        self.M = M
    
        self.x = x
        self.t = t
    
        self.x_phi = np.zeros([N, M+1])
    
        self.w = 0
    
        self.E_D = 0
        self.E_RMS = 0
    
    def w_cal(self):
    
        w_1 = np.dot(self.x_phi.T, self.x_phi)
        w_2 = np.linalg.inv(w_1)
        w_3 = np.dot(w_2, self.x_phi.T)
        w = np.dot(w_3, self.t)
    
        return w

    def f(self, x_f):
    
        f_cal = 0
    
        for m in range(self.M+1):
            f_cal += (self.w[m] * (x_f ** m))
        
        return f_cal
    
    def Execution(self):
    
        for n in range(self.N):
            for m in range(self.M+1):
                self.x_phi[n, m] = self.x[n] ** m
            
        self.w = self.w_cal()
    
        for n in range(self.N):
            self.E_D += (self.f(self.x[n]) - self.t[n]) ** 2
        
        self.E_D = self.E_D / 2
    
        self.E_RMS = np.sqrt(2 * self.E_D / self.N)
        
        return self.E_RMS
    

# -----matplotlib -----

def main():

    N = 100             # データの個数
    M = [0, 1, 3, 9]    # E_D でのwの係数xの冪乗の個数



    x = np.linspace(0, 1, N)
    t = np.sin(2 * np.pi * x) + np.random.normal(loc=0, scale=0.3, size=N)

    x_sin = np.linspace(0, 1, 100)
    t_sin = np.sin(2 * np.pi * x_sin)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    t_0 = 0 # axesの行調節
    t_1 = 0 

    for i in M:
        if t_1 == 2: # 2行目に移行するかの判断
            t_0 += 1
            t_1 = 0
        Lea = Lea_squ_method(N, i, x, t)
        E_RSM = Lea.Execution()
        f_phi = np.zeros([100, i+1])
        for n in range(100):
            for m in range(i+1):
                f_phi[n, m] = x_sin[n] ** m
        f = np.dot(f_phi, Lea.w.reshape(i+1, 1))
        f = np.squeeze(f)
        axes[t_0, t_1].plot(x_sin, t_sin, linestyle="dashed", label="t = sin(2πx)")
        axes[t_0, t_1].scatter(x, t)
        axes[t_0, t_1].plot(x_sin, f, label="E(RSM) = " + str(E_RSM))
        axes[t_0, t_1].set_xlabel("x")
        axes[t_0, t_1].set_ylabel("t")
        axes[t_0, t_1].set_title("M = " + str(i))
        axes[t_0, t_1].legend()
        t_1 += 1
    plt.show()

main()
