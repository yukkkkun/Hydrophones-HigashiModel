import numpy as np
import pandas as pd
import math

from logging import getLogger, StreamHandler, DEBUG, INFO

#自作モジュール
import getdfs

#############
#ログ設定
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False
#############
# Unit Setting
# UNIT = 'm'
UNIT = 'cm'

#################
if UNIT == 'm':
    ##初期値m, kg
    rho = 1000 #kg/m3
    grav = 9.8 #m/s**2
    ib = 1/20 #? パーセント？角度？
    s = 1.65
    width = 5#m
    sigma_by_rho = 2.65
    rho_s = 2650 #kg/m**3
    Rb = s #? 砂礫の水中比重
    W_IDEAL = np.array([0.15, 0.22, 0.29, 0.65, 0.91, 1.96, 3.01, 6.91, 10.81, 50])*0.001 #kg


if UNIT == 'cm':
    ##初期値m, g
    rho = 1 #g/cm3
    grav = 980 #cm/s**2
    ib = 1/20 #? パーセント？角度？
    s = 1.65
    width = 500#m
    sigma_by_rho = 2.65
    rho_s = 2.65 #g/cm**3
    Rb = s #? 砂礫の水中比重
    W_IDEAL = np.array([0.15, 0.22, 0.29, 0.65, 0.91, 1.96, 3.01, 6.91, 10.81, 50]) #g
    D_IDEAL = 2*((W_IDEAL/rho_s)*(3/4)*(1/math.pi))**(1/3)


#################


def calc_R(WaterLevel, width):
    '''
    R=A/S
    A:面積
    S:
    '''
    A = WaterLevel*width
    S = 2*WaterLevel + width

    logger.debug('R : {}'.format(A/S))

    return A/S

def calc_u_star(R):
    
    u_star = np.sqrt(grav*R*ib)
    logger.debug('u_star : {}'.format(u_star))

    return u_star


def calc_u_star_c(d_cm):
    if d_cm >= 0.303:
        u_star_c_2 = 80.9*d_cm
    elif 0.118 <= d_cm < 0.303:
        u_star_c_2 = 134.6*d_cm**(31/22)
    elif 0.0565 <= d_cm < 0.118:
        u_star_c_2 = 55.0*d_cm
    elif 0.0065 <= d_cm < 0.0565:
        u_star_c_2 = 8.41*d_cm**(11/32)
    elif 0 <= d_cm < 0.0065:
        u_star_c_2 = 226*d_cm
    else:
        logger.error('Diameter is out of range.')

    u_star_c = np.sqrt(u_star_c_2)
    logger.debug('u_star_c : {}'.format(u_star_c))

    return u_star_c

def calc_tau_star(R, d):
    u_star = calc_u_star(R)
    tau_star = (u_star**2)/((sigma_by_rho-1)*grav*d)

    logger.debug('tau_star : {}'.format(tau_star))

    return tau_star

def calc_tau_star_c(d):
    if UNIT == 'm':
        d_cm = d*100 #cmに変換
    if UNIT == 'cm':
        d_cm = d
    u_star_c = calc_u_star_c(d_cm)
    tau_star_c = (u_star_c**2)/((sigma_by_rho-1)*grav*d)
    logger.debug('tau_star_c : {}'.format(tau_star_c))

    return tau_star_c

def calc_u_s(R, d):
    tau_star = calc_tau_star(R, d)
    tau_star_c = calc_tau_star_c(d)

    u_s = (1.56*(((tau_star/tau_star_c)-1)**0.56))*np.sqrt(Rb*grav*d)
    logger.debug('u_s : {}'.format(u_s))

    return u_s

def calc_l_s(R, d):
    tau_star = calc_tau_star(R, d)
    tau_star_c = calc_tau_star_c(d)

    l_s = (8.0*((tau_star/tau_star_c)-1)**0.88)*d
    logger.debug('ls : {}'.format(l_s))

    return l_s

def calc_h_s(R, d):
    tau_star = calc_tau_star(R, d)
    tau_star_c = calc_tau_star_c(d)

    h_s = (1.44*((tau_star/tau_star_c)-1)**0.5)*d
    logger.debug('h_s : {}'.format(h_s))

    return h_s


if __name__ == "__main__":

    WaterLevel = 10#cm
    
    logger.info('D_IDEAL(cm) : {}'.format(D_IDEAL))
    R = calc_R(WaterLevel, width)
    logger.info('R : {}'.format(R))

    # diameter = 6 #cm
    for diameter in D_IDEAL:

        u_s = calc_u_s(R, diameter)
        l_s = calc_l_s(R, diameter)
        h_s = calc_h_s(R, diameter)
        print(h_s)
    

    print("Sklar-Dientrich run succecfully")