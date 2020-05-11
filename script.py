# import RK4

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

def func1(X1):
    '''
    dx/dt = y -> func1
    '''
    return X1


def func2(t, y, x, u):
    '''
    dy/dt = u*(1-x^2)*y-x -> func2
    '''
    F2 = u*(1 - x**2)*y - x
    return F2


def main():
    # X = np.empty((N, 1))
    # U = np.empty((M, 1))

    a1 = 0.166
    a23 = 0.333
    # a4 = a1

    dt = 0.01
    TF = 150
    # Condicoes iniciais
    #               y    x
    X = np.array([[0.01, 0]])
    # X = np.array([[9.97983375e-05, 9.99950100e-03]])
    # X[0] = 0.01 # y
    # X[1] = 0 # x
    t = np.array([[0]]).T  # Vetor coluna
    U = np.array([[]]).T  # Vetor coluna

    while t[-1] <= TF:
        U = np.append(U, [np.sin(t[-1]**0.2)], axis=0)

        t2 = t[-1] + 0.5*dt
        t3 = t[-1] + dt

        K1 = dt*func1(X[-1, 0])
        L1 = dt*func2(t, X[-1, 0], X[-1, 1], U[-1])

        K2 = dt*func1(X[-1, 0] + 0.5*L1)
        L2 = dt*func2(t2, X[-1, 0] + 0.5*L1, X[-1, 1] + 0.5*K1, U[-1])

        K3 = dt*func1(X[-1, 0] + 0.5*L2)
        L3 = dt*func2(t2, X[-1, 0] + 0.5*L2, X[-1, 1] + 0.5*K2, U[-1])

        K4 = dt*func1(X[-1, 0] + L3)
        L4 = dt*func2(t3, X[-1, 0] + L3, X[-1, 1] + K3, U[-1])

        # Update X-Position
        X1 = X[-1, 1] + a1*(K1 + K4) + a23*(K2 + K3)  
        X0 = X[-1, 0] + a1*(L1 + L4) + a23*(L2 + L3)

        X = np.append(X, np.array([X0, X1]).T, axis=0)
        # f.write('{:.8e} , {} , {} , {} \n'.format(
        #     t, X[1], X[0], U[-1]))
        t = np.append(t, [t[-1] + dt], axis=0)  # Incrementando o timestep

    print('t', t.shape)
    print('X', X.shape)
    print('U', U.shape)
    matriz = np.concatenate((t[0:-1], X[0:-1,[1,0]], U), axis=1)

    pd.DataFrame(matriz).to_csv('sim.csv', header=[
        'Time(s)', 'X-Position (X[1])', 'Y-Position (X[0])', 'Input U'], float_format='%.8e')

    print('\n[X[1] X[0]] = {}'.format(X[-1, :]))
    print('t = {}\n'.format(t[-1]))


def graficos():
    sim_data = pd.read_csv('sim.csv')
    # print(sim_data['Time(s)'].unique())
    # sim_data_red = sim_data.head(200)
    # print(sim_data_red)
    fig = plt.figure( figsize=(12, 4))
    plt.title('Van der Pol 2D oscillator')
    plt.subplot(2, 2, 1)
    plt.plot(sim_data['Time(s)'], sim_data['X-Position (X[1])'])
    plt.xlabel('Time(s)')
    plt.ylabel('X-Position (m)')
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(sim_data['Time(s)'],
             sim_data['Y-Position (X[0])'])
    plt.xlabel('Time(s)')
    plt.ylabel('Y-Position (m)')
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.plot(sim_data['X-Position (X[1])'],
             sim_data['Y-Position (X[0])'])
    plt.xlabel('X-Position (m)')
    plt.ylabel('Y-Position (m)')
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.plot(sim_data['Time(s)'],
             sim_data['Input U'])
    plt.xlabel('Time(s)')
    plt.ylabel('Input U')
    plt.grid()
    plt.subplots_adjust(wspace=0.45, hspace=0.50)
    plt.show()


# main()
graficos()
