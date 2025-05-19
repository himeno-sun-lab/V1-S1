import matplotlib.pyplot as plt
import numpy as np

def create_circle(center, rad, col):
    circle = plt.Circle((center[0],center[1]), radius = rad, color = col, fill=False, linewidth=4)
    return circle

#plt.rcParams ["figure.figsize"] = [32, 32]
circle_m = create_circle((0,0), .16, 'r')
circle0 = create_circle((0,0), .05, 'b')
circle1 = create_circle((0.11 * np.cos( np.pi / 6. ), 0.11 * np.sin( np.pi / 6. )), .05, 'b')
circle2 = create_circle((0,0.11), .05, 'b')
circle3 = create_circle((-0.11 * np.cos( np.pi / 6. ), 0.11 * np.sin( np.pi / 6. )), .05, 'b')
circle4 = create_circle((-0.11 * np.cos( np.pi / 6. ), -0.11 * np.sin( np.pi / 6. )), .05, 'b')
circle5 = create_circle((0,-0.11), .05, 'b')
circle6 = create_circle((0.11 * np.cos( np.pi / 6. ), -0.11 * np.sin( np.pi / 6. )), .05, 'b')

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='datalim')
ax.add_artist(circle_m)
ax.add_artist(circle0)
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
ax.add_artist(circle5)
ax.add_artist(circle6)
ax.text(-0.02,-0.02, '0', fontsize = 40)
ax.text(0.11 * np.cos( np.pi / 6. )-0.02, 0.11 * np.sin( np.pi / 6. )-0.02, '1', fontsize = 40)
ax.text(-0.02, 0.11-0.02, '2', fontsize = 40)
ax.text(-0.11 * np.cos( np.pi / 6. )-0.02, 0.11 * np.sin( np.pi / 6. )-0.02, '3', fontsize = 40)
ax.text(-0.11 * np.cos( np.pi / 6. )-0.02, -0.11 * np.sin( np.pi / 6. )-0.02, '4', fontsize = 40)
ax.text(-0.02, -0.11-0.02, '5', fontsize = 40)
ax.text(0.11 * np.cos( np.pi / 6. )-0.02, -0.11 * np.sin( np.pi / 6. )-0.02, '6', fontsize = 40)
plt.xlim( -0.3, 0.3 )
plt.ylim( -0.3, 0.3 )
plt.grid()
plt.show()
save_results_to = '/home/morteza/Desktop/reports/191105 (micro_col_first_test)/'
plt.savefig( save_results_to + 'columns.jpeg', bbox_inches='tight' )

circle_center = [[0.0, 0.0], [0.11 * np.cos( np.pi / 6. ), 0.11 * np.sin( np.pi / 6. )], [0.0, 0.11],
                 [-0.11 * np.cos( np.pi / 6. ), 0.11 * np.sin( np.pi / 6. )],
                 [-0.11 * np.cos( np.pi / 6. ), -0.11 * np.sin( np.pi / 6. )], [0.0, -0.11],
                 [0.11 * np.cos( np.pi / 6. ), -0.11 * np.sin( np.pi / 6. )]]