import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']        # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False          # 让负号正常显示

plt.plot([1, 2, 3], [4, 5, 6])
plt.title('示例：中文标题')
plt.xlabel('横轴标签')
plt.ylabel('纵轴标签')
plt.show()