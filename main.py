# simulation code from: https://github.com/mpkuse/inverted_pendulum.git

import cv2
import numpy as np
from InvertedPendulum import InvertedPendulum as IP
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

angle = ctrl.Antecedent(np.arange(-15, 15, 1), 'angle') # base
angular_vel = ctrl.Antecedent(np.arange(-6, 6, 1), 'angular_vel') # base
force = ctrl.Consequent(np.arange(-100, 100, 1), 'force') # base

angle.automf(7)
angular_vel.automf(7)
force.automf(7)

rule01 = ctrl.Rule(angle['dismal']& angular_vel['dismal'], force['dismal'])
rule02 = ctrl.Rule(angle['dismal']& angular_vel['poor'], force['dismal'])
rule03 = ctrl.Rule(angle['dismal']& angular_vel['mediocre'], force['dismal'])
rule04 = ctrl.Rule(angle['dismal']& angular_vel['average'], force['dismal'])
rule05 = ctrl.Rule(angle['dismal']& angular_vel['decent'], force['poor'])
rule06 = ctrl.Rule(angle['dismal']& angular_vel['good'], force['mediocre'])
rule07 = ctrl.Rule(angle['dismal']& angular_vel['excellent'], force['decent'])

rule08 = ctrl.Rule(angle['poor']& angular_vel['dismal'], force['dismal'])
rule09 = ctrl.Rule(angle['poor']& angular_vel['poor'], force['dismal'])
rule10 = ctrl.Rule(angle['poor']& angular_vel['mediocre'], force['poor'])
rule11 = ctrl.Rule(angle['poor']& angular_vel['average'], force['poor'])
rule12 = ctrl.Rule(angle['poor']& angular_vel['decent'], force['mediocre'])
rule13 = ctrl.Rule(angle['poor']& angular_vel['good'], force['decent'])
rule14 = ctrl.Rule(angle['poor']& angular_vel['excellent'], force['decent'])

rule15 = ctrl.Rule(angle['mediocre']& angular_vel['dismal'], force['dismal'])
rule16 = ctrl.Rule(angle['mediocre']& angular_vel['poor'], force['poor'])
rule17 = ctrl.Rule(angle['mediocre']& angular_vel['mediocre'], force['mediocre'])
rule18 = ctrl.Rule(angle['mediocre']& angular_vel['average'], force['mediocre'])
rule19 = ctrl.Rule(angle['mediocre']& angular_vel['decent'], force['decent'])
rule20 = ctrl.Rule(angle['mediocre']& angular_vel['good'], force['decent'])
rule21 = ctrl.Rule(angle['mediocre']& angular_vel['excellent'], force['good'])

rule22 = ctrl.Rule(angle['average']& angular_vel['dismal'], force['dismal'])
rule23 = ctrl.Rule(angle['average']& angular_vel['poor'], force['poor'])
rule24 = ctrl.Rule(angle['average']& angular_vel['mediocre'], force['mediocre'])
rule25 = ctrl.Rule(angle['average']& angular_vel['average'], force['average'])
rule26 = ctrl.Rule(angle['average']& angular_vel['decent'], force['decent'])
rule27 = ctrl.Rule(angle['average']& angular_vel['good'], force['good'])
rule28 = ctrl.Rule(angle['average']& angular_vel['excellent'], force['excellent'])

rule29 = ctrl.Rule(angle['decent']& angular_vel['dismal'], force['poor'])
rule30 = ctrl.Rule(angle['decent']& angular_vel['poor'], force['mediocre'])
rule31 = ctrl.Rule(angle['decent']& angular_vel['mediocre'], force['mediocre'])
rule32 = ctrl.Rule(angle['decent']& angular_vel['average'], force['decent'])
rule33 = ctrl.Rule(angle['decent']& angular_vel['decent'], force['decent'])
rule34 = ctrl.Rule(angle['decent']& angular_vel['good'], force['good'])
rule35 = ctrl.Rule(angle['decent']& angular_vel['excellent'], force['excellent'])

rule36 = ctrl.Rule(angle['good']& angular_vel['dismal'], force['mediocre'])
rule37 = ctrl.Rule(angle['good']& angular_vel['poor'], force['mediocre'])
rule38 = ctrl.Rule(angle['good']& angular_vel['mediocre'], force['decent'])
rule39 = ctrl.Rule(angle['good']& angular_vel['average'], force['good'])
rule40 = ctrl.Rule(angle['good']& angular_vel['decent'], force['good'])
rule41 = ctrl.Rule(angle['good']& angular_vel['good'], force['excellent'])
rule42 = ctrl.Rule(angle['good']& angular_vel['excellent'], force['excellent'])

rule43 = ctrl.Rule(angle['excellent']& angular_vel['dismal'], force['mediocre'])
rule44 = ctrl.Rule(angle['excellent']& angular_vel['poor'], force['decent'])
rule45 = ctrl.Rule(angle['excellent']& angular_vel['mediocre'], force['good'])
rule46 = ctrl.Rule(angle['excellent']& angular_vel['average'], force['excellent'])
rule47 = ctrl.Rule(angle['excellent']& angular_vel['decent'], force['excellent'])
rule48 = ctrl.Rule(angle['excellent']& angular_vel['good'], force['excellent'])
rule49 = ctrl.Rule(angle['excellent']& angular_vel['excellent'], force['excellent'])

force_cntrl = ctrl.ControlSystem([
    rule01, rule02, rule03, rule04, rule05, rule06, rule07,
    rule08, rule09, rule10, rule11, rule12, rule13, rule14,
    rule15, rule16, rule17, rule18, rule19, rule20, rule21,
    rule22, rule23, rule24, rule25, rule26, rule27, rule28,
    rule29, rule30, rule31, rule32, rule33, rule34, rule35,
    rule36, rule37, rule38, rule39, rule40, rule41, rule42,
    rule43, rule44, rule45, rule46, rule47, rule48, rule49
])

fuzzy_force = ctrl.ControlSystemSimulation(force_cntrl) 

def uforce(statevec): 
    global fuzzy_force
    fuzzy_force.input['angle'] = (statevec[2]*180. / np.pi) - 90
    fuzzy_force.input['angular_vel'] = statevec[3]
    fuzzy_force.compute()
    return fuzzy_force.output['force'] 


def func2(t, y):
    g = 9.8 # Gravitational Acceleration
    L = 1.5 # Length of pendulum
    m = 1.0 #mass of bob (kg)
    M = 2.0  # mass of cart (kg)
    x_ddot = uforce(y) - m*L*y[3]*y[3] * np.cos( y[2] ) + m*g*np.cos(y[2]) *  np.sin(y[2])
    x_ddot = x_ddot / ( M+m-m* np.sin(y[2])* np.sin(y[2]) )
    theta_ddot = -g/L * np.cos( y[2] ) - 1./L * np.sin( y[2] ) * x_ddot
    damping_theta =  - 0.5*y[3]
    damping_x =  - 1.0*y[1]
    return [ y[1], x_ddot + damping_x, y[3], theta_ddot + damping_theta ]


def simulation(statevec, t, frames):
    return solve_ivp(func2, [0, t], statevec, t_eval=np.linspace( 0, t, frames)  ) 


def display_simulation(sol): 
    model = IP()
    for i, t in enumerate(sol.t): 
        render = model.step( [sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i] ], t )
        cv2.imshow('simulation', render)
        if cv2.waitKey(15) == ord('q'):
            break
    

def run(iter, time, frames): 
    # run initial simulation
    result = simulation(statevec, time, frames) # medio segundo 30 frames
    new_statevec = [
        round(result.y[0,frames-1], 2), round(result.y[1,frames-1], 2),
        round(result.y[2,frames-1], 2), round(result.y[3,frames-1],2) ]
    display_simulation(result)

    #save initial stats
    stats = [0, ((np.pi/2 + 0.5)*180/np.pi)-90, 0, 0] 
    stat_angle = (new_statevec[2]*180. / np.pi) - 90 
    stat_angular_vel = new_statevec[3]
    stats = np.vstack([ stats, [0.5, round(stat_angle,2), round(stat_angular_vel,2), 0] ])
    for i in range(iter): 
        # run simulation 
        aux = np.asarray(new_statevec)
        stat_force = uforce(aux)
        result = simulation(aux, time, frames) # medio segundo 30 frames
        new_statevec = [result.y[0,frames-1], result.y[1,frames-1], result.y[2,frames-1], result.y[3,frames-1] ]
        display_simulation(result)

        #save states
        stat_angle = ((new_statevec[2]*180. / np.pi) - 90 )
        stat_angular_vel = new_statevec[3]
        stats = np.vstack([ stats, [0.5*(i+2), round(stat_angle, 2), round(stat_angular_vel,2), round(stat_force, 2)] ])
    return stats


statevec = [-1, 0., np.pi/2 + 0.5, 0]

stats = run(10, 0.25, 30)
print(stats)
fig = plt.figure()
ax = plt.axes(xlim=(0, 5.5), ylim=(-15, 15))
angle.view()
angular_vel.view()
force.view()

angle, = ax.plot(stats[:,0], stats[:, 1], lw=2, label='angle')
angular = ax.plot(stats[:,0], stats[:, 2], lw=2, label='angular velocity')
forces = ax.plot(stats[:,0], stats[:, 3], lw=2, label='force')
plt.show()