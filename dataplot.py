from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def path_classification_plot(states_info):
    x = {}
    y = {}
    z = {}
    fig = plt.figure(2)
    ax = Axes3D(fig, auto_add_to_figure=False)
    for key in states_info:
        position3d = np.array(list(states_info[key]))
        x[key] = position3d[:, 0]
        y[key] = position3d[:, 1]
        z[key] = position3d[:, 2]
        if key == 0:
            ax.scatter(np.array(list(x[0])), np.array(list(y[0])), np.array(list(z[0])), zdir="z", c="#00DDAA", marker="o", s=40)
        elif key == 1:
            ax.scatter(np.array(list(x[1])), np.array(list(y[1])), np.array(list(z[1])), zdir="z", c="#FF5511", marker="o", s=40)
        elif key == 2:
            ax.scatter(np.array(list(x[2])), np.array(list(y[2])), np.array(list(z[2])), zdir="z", c="#0000CD", marker="o", s=40)
        elif key == 3:
            ax.scatter(np.array(list(x[3])), np.array(list(y[3])), np.array(list(z[3])), zdir="z", c="#FFA500", marker="o", s=40)
        elif key == 4:
            ax.scatter(np.array(list(x[4])), np.array(list(y[4])), np.array(list(z[4])), zdir="z", c="#C0C0C0", marker="o", s=40)
        elif key == 5:
            ax.scatter(np.array(list(x[5])), np.array(list(y[5])), np.array(list(z[5])), zdir="z", c="#FFD700", marker="o", s=40)

    ax.set(xlabel="X(0~1)", ylabel="Y(0~1)", zlabel="V")
    ax.set_title(label="after first classification")
    fig.add_axes(ax)
    plt.show()


def gaze_trace_xyv_plot(datas):
    x = datas[:, 0]
    y = datas[:, 1]
    z = datas[:, 2]
    fig = plt.figure(1)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.scatter(x, y, z, zdir="z", c="#000000", marker="o", s=40)
    fig.add_axes(ax)
    ax.set(xlabel="X(0~1)", ylabel="Y(0~1)", zlabel="V")
    ax.set_title(label="raw datas", fontproperties="SimHei", fontsize=24)
    plt.show()


def gaze_point_classification_plot(states_info, gaze_states_info, num):
    fig = plt.figure(3)
    ax = Axes3D(fig, auto_add_to_figure=False)
    x = {}
    y = {}
    z = {}
    for key in states_info:
        position3d = np.array(list(states_info[key]))
        x[key] = position3d[:, 0]
        y[key] = position3d[:, 1]
        z[key] = position3d[:, 2]
    for key in x:
        x1 = np.array(list(x[key]))
        y1 = np.array(list(y[key]))
        z1 = np.array(list(z[key]))
        index_0 = []
        index_1 = []
        index_2 = []
        for index, i in enumerate(np.array(list(gaze_states_info[key]))):
            if i == 0:
                index_0.append(index)
            elif i == 1:
                index_1.append(index)
            elif i == 2:
                index_2.append(index)
            x_0 = x1[index_0]
            y_0 = y1[index_0]
            z_0 = z1[index_0]
            x_1 = x1[index_1]
            y_1 = y1[index_1]
            z_1 = z1[index_1]
            x_2 = x1[index_2]
            y_2 = y1[index_2]
            z_2 = z1[index_2]
        if key == 0:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#00DDAA", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#00DDAA", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#00DDAA", marker=".", s=40)
        elif key == 1:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#FF5511", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#FF5511", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#FF5511", marker=".", s=40)
        elif key == 2:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#0000CD", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#0000CD", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#0000CD", marker=".", s=40)
        elif key == 3:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#FFA500", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#FFA500", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#FFA500", marker=".", s=40)
        elif key == 4:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#C0C0C0", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#C0C0C0", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#C0C0C0", marker=".", s=40)
        elif key == 5:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#FFD700", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#FFD700", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#FFD700", marker=".", s=40)
    fig.add_axes(ax)
    ax.set(xlabel="X(0~1)", ylabel="Y(0~1)", zlabel="V")
    plt.show()


def gazepoint_classification_final_results(states_info, gaze_states_info):
    fig = plt.figure(4)
    ax = Axes3D(fig, auto_add_to_figure=False)
    x = {}
    y = {}
    z = {}
    result = {}
    for key in states_info:
        position3d = np.array(list(states_info[key]))
        x[key] = position3d[:, 0]
        y[key] = position3d[:, 1]
        z[key] = position3d[:, 2]
    for key in x:
        x1 = np.array(list(x[key]))
        y1 = np.array(list(y[key]))
        z1 = np.array(list(z[key]))
        # 存放索引
        index_0 = []
        index_1 = []
        index_2 = []
        index_ternary = np.zeros(len(x1))
        for index, i in enumerate(np.array(list(gaze_states_info[key]))):
            if i == 0:
                index_0.append(index)
            elif i == 1:
                index_1.append(index)
            elif i == 2:
                index_2.append(index)
        index_fx = []
        index_sp = []
        index_sc = []
        if z1[index_0[0]] > z1[index_1[0]]:            #a>b
            if z1[index_1[0]] > z1[index_2[0]]:        # a>b>c
                index_fx = index_2
                index_sp = index_1
                index_sc = index_0
            elif z1[index_1[0]] < z1[index_2[0]]:     # a>b b<c
                if z1[index_0[0]] > z1[index_2[0]]:  # a>c>b
                    index_fx = index_1
                    index_sp = index_2
                    index_sc = index_0
                elif z1[index_0[0]] < z1[index_2[0]]: # c>a>b
                    index_fx = index_1
                    index_sp = index_2
                    index_sc = index_0
        elif z1[index_0[0]] < z1[index_1[0]]:      # a<b
            if z1[index_1[0]] < z1[index_2[0]]:    # a<b<c
                index_fx = index_0
                index_sp = index_1
                index_sc = index_2
            elif z1[index_1[0]] > z1[index_2[0]]:  # b>c
                if z1[index_0[0]] < z1[index_2[0]]:  # a<c<b
                    index_fx = index_0
                    index_sp = index_2
                    index_sc = index_1
                elif z1[index_0[0]] > z1[index_2[0]]: # a>b>c
                    index_fx = index_2
                    index_sp = index_1
                    index_sc = index_0
        x_0 = x1[index_fx]
        y_0 = y1[index_fx]
        z_0 = z1[index_fx]
        x_1 = x1[index_sp]
        y_1 = y1[index_sp]
        z_1 = z1[index_sp]
        x_2 = x1[index_sc]
        y_2 = y1[index_sc]
        z_2 = z1[index_sc]
        index_ternary[index_fx] = 0
        index_ternary[index_sp] = 1
        index_ternary[index_sc] = 2
        result[key] = list(index_ternary)
        ax.scatter(x_0, y_0, z_0, zdir="z", c="#00DDAA", marker="x", s=40)
        ax.scatter(x_1, y_1, z_1, zdir="z", c="#FF5511", marker="^", s=40)
        ax.scatter(x_2, y_2, z_2, zdir="z", c="#0000CD", marker=".", s=40)
    fig.add_axes(ax)
    ax.set(xlabel="X(0~1)", ylabel="Y(0~1)", zlabel="V")
    plt.show()
    return result
