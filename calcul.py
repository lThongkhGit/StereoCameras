import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import pandas as pd
import matplotlib.animation as animation

'''
A script to test out some code lines
'''

# Fixing random state for reproducibility
np.random.seed(19680801)

df_wrist = pd.read_csv("Data/wrist.csv")
df_thumb_tip = pd.read_csv("Data/thumbs_tip.csv")
df_index_mcp = pd.read_csv("Data/index_mcp.csv")
df_pinky_mcp = pd.read_csv("Data/pinky_mcp.csv")
print(df_wrist.head())

x = []
y = []
z = []
pos_wrist = []
pos_thumbs_cmc = []
pos_index_mcp = []
pos_pinky_mcp = []
for i in range(len(df_wrist['1'])):
    x.append(df_wrist['0'][i])
    x.append(df_thumb_tip['0'][i])
    x.append(df_index_mcp['0'][i])
    x.append(df_pinky_mcp['0'][i])
    y.append(df_wrist['1'][i])
    y.append(df_thumb_tip['1'][i])
    y.append(df_index_mcp['1'][i])
    y.append(df_pinky_mcp['1'][i])
    z.append(df_wrist['2'][i])
    z.append(df_thumb_tip['2'][i])
    z.append(df_index_mcp['2'][i])
    z.append(df_pinky_mcp['2'][i])
    pos_wrist.append([df_wrist['0'][i], df_wrist['1'][i], df_wrist['2'][i]])
    pos_thumbs_cmc.append([df_thumb_tip['0'][i], df_thumb_tip['1'][i], df_thumb_tip['2'][i]])
    pos_index_mcp.append([df_index_mcp['0'][i], df_index_mcp['1'][i], df_index_mcp['2'][i]])
    pos_pinky_mcp.append([df_pinky_mcp['0'][i], df_pinky_mcp['1'][i], df_pinky_mcp['2'][i]])

pos = [pos_wrist, pos_thumbs_cmc, pos_index_mcp, pos_pinky_mcp]

a = np.random.rand(2000, 3)*10
#t = np.array([np.ones(100) * i for i in range(20)]).flatten()
#df = pd.DataFrame({"time": t ,"x" : a[:,0], "y" : a[:,1], "z" : a[:,2]})
t = np.array([np.ones(4) * i for i in range(len(df_wrist['0']))]).flatten()
df = pd.DataFrame({"time": t ,"x" : x, "y" : y, "z" : z})

def update_graph(num):
    data=df[df['time']==num]
    graph._offsets3d = (data.x, data.y, data.z)
    '''for line in lines:
        line.set_data_3d(np.array([pos_wrist[num], pos_thumbs_cmc[num]]).T)'''
    lines[0].set_data_3d(np.array([pos_wrist[num], pos_thumbs_cmc[num]]).T)
    lines[1].set_data_3d(np.array([pos_wrist[num], pos_index_mcp[num]]).T)
    lines[2].set_data_3d(np.array([pos_index_mcp[num], pos_pinky_mcp[num]]).T)
    lines[3].set_data_3d(np.array([pos_wrist[num], pos_pinky_mcp[num]]).T)
    title.set_text('3D Test, time={}'.format(num))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

lines = [ax.plot([], [], [])[0] for i in range(4)]

ax.set_xlim3d([-30.0, 30.0])
ax.set_xlabel('X')
ax.set_ylim3d([-10.0, 30.0])
ax.set_ylabel('Y')
ax.set_zlim3d([0.0, 10.0])
ax.set_zlabel('Z')

data=df[df['time']==0]
graph = ax.scatter(data.x, data.y, data.z)

ani = matplotlib.animation.FuncAnimation(fig, update_graph, len(df_wrist['0']), # fargs=(pos, lines),
                               interval=40, blit=False)

writervideo = animation.PillowWriter(fps=60)
ani.save('increasingStraightLine.gif', writer=writervideo)
plt.show()