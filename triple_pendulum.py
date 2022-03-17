import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(dv, ti, p):
	th1, w1, th2, w2, th3, w3 = dv
	m1, l1, m2, l2, m3, l3, gc = p

	print(ti)
	
	return [w1,alpha1.subs({M1:m1, M2:m2, M3:m3, L1:l1, L2:l2, L3:l3, theta1:th1, theta2:th2, theta3:th3, theta1_dot:w1, theta2_dot:w2, theta3_dot:w3, g:gc}),\
		w2,alpha2.subs({M1:m1, M2:m2, M3:m3, L1:l1, L2:l2, L3:l3, theta1:th1, theta2:th2, theta3:th3, theta1_dot:w1, theta2_dot:w2, theta3_dot:w3, g:gc}),\
		w3,alpha3.subs({M1:m1, M2:m2, M3:m3, L1:l1, L2:l2, L3:l3, theta1:th1, theta2:th2, theta3:th3, theta1_dot:w1, theta2_dot:w2, theta3_dot:w3, g:gc})]

#---SymPy Derivation------------------------

L1,M1,L2,M2,L3,M3,g,t = sp.symbols('L1 M1 L2 M2 L3, M3, g, t')
theta1,theta2,theta3 = dynamicsymbols('theta1 theta2 theta3')

X1 = L1 * sp.sin(theta1)
Y1 = -L1 * sp.cos(theta1)
X2 = X1 + L2 * sp.sin(theta2)
Y2 = Y1 - L2 * sp.cos(theta2)
X3 = X2 + L3 * sp.sin(theta3)
Y3 = Y2 - L3 * sp.cos(theta3)

v1_squared = X1.diff(t, 1)**2 + Y1.diff(t, 1)**2
v2_squared = X2.diff(t, 1)**2 + Y2.diff(t, 1)**2
v3_squared = X3.diff(t, 1)**2 + Y3.diff(t, 1)**2

T = sp.simplify(0.5 *(M1 * v1_squared + M2 * v2_squared + M3 * v3_squared))
V = g * (M1 * Y1 + M2 * Y2 + M3 * Y3)

Lg = T - V

theta1_dot = theta1.diff(t, 1)
dLdtheta1 = Lg.diff(theta1, 1)
dLdtheta1_dot = Lg.diff(theta1_dot, 1)
ddtdLdtheta1_dot = dLdtheta1_dot.diff(t, 1)

theta2_dot = theta2.diff(t, 1)
dLdtheta2 = Lg.diff(theta2, 1)
dLdtheta2_dot = Lg.diff(theta2_dot, 1)
ddtdLdtheta2_dot = dLdtheta2_dot.diff(t, 1)

theta3_dot = theta3.diff(t, 1)
dLdtheta3 = Lg.diff(theta3, 1)
dLdtheta3_dot = Lg.diff(theta3_dot, 1)
ddtdLdtheta3_dot = dLdtheta3_dot.diff(t, 1)

diff_Lg1 = ddtdLdtheta1_dot - dLdtheta1
diff_Lg2 = ddtdLdtheta2_dot - dLdtheta2
diff_Lg3 = ddtdLdtheta3_dot - dLdtheta3

theta1_ddot = theta1.diff(t, 2)
theta2_ddot = theta2.diff(t, 2)
theta3_ddot = theta3.diff(t, 2)
alpha = sp.solve([diff_Lg1, diff_Lg2, diff_Lg3], (theta1_ddot, theta2_ddot, theta3_ddot))


alpha1=sp.factor(sp.simplify(alpha[theta1_ddot]))
alpha2=sp.factor(sp.simplify(alpha[theta2_ddot]))
alpha3=sp.factor(sp.simplify(alpha[theta3_ddot]))

#--------------------------------------------

#---functional working variables we will-----
#---use to sunbstitute into our abstract-----
#---SymPy derivation so that we can----------
#---integrate our differential equation.-----

gc = 9.8
m1,m2,m3 = [1, 1, 1]
l1,l2,l3 = [0.5, 0.75, 1]
theta1_0,theta2_0,theta3_0 = [90, 135, 180] 
theta1_0 *= np.pi/180
theta2_0 *= np.pi/180
theta3_0 *= np.pi/180
omega1_0,omega2_0,omega3_0 = [0, 0, 0]

p = m1, l1, m2, l2, m3, l3, gc
dyn_var = theta1_0, omega1_0, theta2_0, omega2_0, theta3_0, omega3_0

tf = 60 
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

thw = odeint(integrate, dyn_var, ta, args = (p,))

x1=np.zeros(nframes)
y1=np.zeros(nframes)
x2=np.zeros(nframes)
y2=np.zeros(nframes)
x3=np.zeros(nframes)
y3=np.zeros(nframes)
for i in range(nframes):
	x1[i]=X1.subs({L1:l1, theta1:thw[i,0]})
	y1[i]=Y1.subs({L1:l1, theta1:thw[i,0]})
	x2[i]=X2.subs({X1:x1[i], L2:l2, theta2:thw[i,2]})
	y2[i]=Y2.subs({Y1:y1[i], L2:l2, theta2:thw[i,2]})
	x3[i]=X3.subs({X2:x2[i], L3:l3, theta3:thw[i,4]})
	y3[i]=Y3.subs({Y2:y2[i], L3:l3, theta3:thw[i,4]})

ke=np.zeros(nframes)
pe=np.zeros(nframes)
for i in range(nframes):
	ke[i]=T.subs({M1:m1, M2:m2, M3:m3, L1:l1, L2:l2, L3:l3, theta1:thw[i,0], theta2:thw[i,2], theta3:thw[i,4], theta1_dot:thw[i,1], theta2_dot:thw[i,3], theta3_dot:thw[i,5]})
	pe[i]=V.subs({M1:m1, M2:m2, M3:m3, Y1:y1[i], Y2:y2[i], Y3:y3[i], g:gc})
E=ke+pe
Emax=max(E)
E/=Emax
ke/=Emax
pe/=Emax


#--aesthetics-------------------------------

xmax,ymax=(l1+l2+l3)*np.array([1.2, 1.2])
xmin,ymin=(l1+l2+l3)*np.array([-1.2, -1.2])
rad=0.05
phi1=np.arccos(x1/l1)
dx1=rad*np.cos(phi1)
dy1=np.sign(y1)*rad*np.sin(phi1)
dx21=rad*np.sin(thw[:,2])
dy21=rad*np.cos(thw[:,2])
phi2=np.arccos((x2-x1)/l2)
dx22=rad*np.cos(phi2)
dy22=np.sign(y2-y1)*rad*np.sin(phi2)
dx32=rad*np.sin(thw[:,4])
dy32=rad*np.cos(thw[:,4])
phi3=np.arccos((x3-x2)/l3)
dx33=rad*np.cos(phi3)
dy33=np.sign(y3-y2)*rad*np.sin(phi3)

#--plot/animation---------------------------

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	plt.plot([0,x1[frame]-dx1[frame]],[0,y1[frame]-dy1[frame]],color='xkcd:cerulean')
	circle=plt.Circle((x1[frame],y1[frame]),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.plot([x1[frame]+dx21[frame],x2[frame]-dx22[frame]],[y1[frame]-dy21[frame],y2[frame]-dy22[frame]],color='xkcd:cerulean')
	circle=plt.Circle((x2[frame],y2[frame]),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.plot([x2[frame]+dx32[frame],x3[frame]-dx33[frame]],[y2[frame]-dy32[frame],y3[frame]-dy33[frame]],color='xkcd:cerulean')
	circle=plt.Circle((x3[frame],y3[frame]),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.title("The Triple Pendulum")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=0.5)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=0.5)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy (Rescaled)")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('triple_pendulum.mp4', writer=writervideo)
#plt.show()


