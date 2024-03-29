# # import matplotlib.pyplot as plt

# # # 定义你提供的数据
# # data = """
# # PC13, 1, 0,
# # PA0, 0, 0,
# # PA0, 1, 0,
# # PC13, 1, 1,
# # PC13, 1, 0,
# # PA0, 0, 0,
# # PA0, 1, 0,
# # PC13, 1, 1,
# # PC13, 1, 0,
# # PA0, 0, 0,
# # PC13, 0, 1,
# # PA0, 1, 1,
# # """

# # # 解析数据
# # lines = data.strip().splitlines()
# # trigger, pa0, pc13 = [], [], []

# # for line in lines:
# #     parts = line.split(',')
# #     trigger.append(parts[0].strip())
# #     pa0.append(int(parts[1].strip()))
# #     pc13.append(int(parts[2].strip()))

# # # 将触发源转换为数值
# # trigger_values = [1 if t == 'PA0' else 0 for t in trigger]

# # # 创建时间轴
# # time = list(range(len(trigger)))

# # # 绘制波形图
# # plt.figure(figsize=(10, 5))

# # plt.plot(time, pa0, label='PA0', drawstyle='steps-post')
# # plt.plot(time, pc13, label='PC13', drawstyle='steps-post')
# # # plt.plot(time, trigger_values, label='Trigger (PA0=1, PC13=0)', linestyle='--')

# # plt.xlabel('Time')
# # plt.ylabel('Level')
# # plt.ylim(-0.5, 1.5)
# # plt.title('Waveform Plot')
# # plt.legend(loc='upper left')
# # plt.grid(True)

# # # 显示波形图
# # plt.show()

# import matplotlib.pyplot as plt

# # 定义你提供的数据
# data = """
# PC13, 1, 0,
# PA0, 0, 0,
# PA0, 1, 0,
# PC13, 1, 1,
# PC13, 1, 0,
# PA0, 0, 0,
# PA0, 1, 0,
# PC13, 1, 1,
# PC13, 1, 0,
# PA0, 0, 0,
# PC13, 0, 1,
# PA0, 1, 1,
# """

# # 解析数据
# lines = data.strip().splitlines()
# trigger, pa0, pc13 = [], [], []

# for line in lines:
#     parts = line.split(',')
#     trigger.append(parts[0].strip())
#     pa0.append(int(parts[1].strip()))
#     pc13.append(int(parts[2].strip()))

# # 创建时间轴
# time = list(range(len(trigger)))

# # 设置图表大小
# plt.figure(figsize=(10, 6))

# # 绘制 PA0 的波形图
# plt.subplot(2, 1, 1)  # 2行1列，第1个图
# plt.plot(time, pa0, label='PA0', drawstyle='steps-post')
# plt.ylabel('Level')
# plt.ylim(-0.5, 1.5)
# plt.title('PA0 Waveform')
# plt.legend(loc='upper left')
# plt.grid(True)

# # 绘制 PC13 的波形图
# plt.subplot(2, 1, 2)  # 2行1列，第2个图
# plt.plot(time, pc13, label='PC13', drawstyle='steps-post', color='orange')
# plt.xlabel('Time')
# plt.ylabel('Level')
# plt.ylim(-0.5, 1.5)
# plt.title('PC13 Waveform')
# plt.legend(loc='upper left')
# plt.grid(True)

# # 调整子图间的间隔
# plt.tight_layout()

# # 显示波形图
# plt.show()


import matplotlib.pyplot as plt

# 定义你提供的数据
data = """
PA0,0,1,
PC13,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PC13,0,1,
PA0,1,1,
PA0,0,1,
PC13,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PC13,0,1,
PA0,1,1,
PA0,0,1,
PC13,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PC13,0,1,
PC13,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PC13,0,1,
PA0,1,1,
PC13,1,0,
PA0,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PC13,0,1,
PA0,1,1,
PC13,1,0,
PA0,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PC13,0,1,
PC13,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PC13,0,1,
PC13,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PC13,0,1,
PC13,0,0,
PA0,1,0,
PC13,1,1,
PC13,1,0,
PA0,0,0,
PC13,0,1,
PA0,1,1,
"""

# 解析数据
lines = data.strip().splitlines()
triggers, pa0, pc13 = [], [], []

for line in lines:
    parts = line.split(",")
    triggers.append(parts[0].strip())
    pa0.append(int(parts[1].strip()) * 2)  # 放大PA0信号，使其在上方显示
    pc13.append(int(parts[2].strip()))  # PC13信号保持在下方

# 创建时间轴
time = list(range(len(pa0)))

# 设置图表大小
plt.figure(figsize=(len(pc13), 2))

# 绘制波形图
plt.plot(time, pa0, drawstyle="steps-post", label="PA0")
plt.plot(time, pc13, drawstyle="steps-post", label="PC13")

# for i, t in enumerate(trigger):
#     if t == 'PA0':
#         plt.axvline(x=i, color='red', linestyle='--')  # PA0触发的地方
#     elif t == 'PC13':
#         plt.axvline(x=i, color='blue', linestyle='--')  # PC13触发的地方

# for i in range(1, len(pa0)):
#     if pa0[i] != pa0[i - 1]:  # PA0 电平变化
#         plt.annotate(
#             "",
#             xy=(i, pa0[i]),
#             xytext=(i, pa0[i - 1]),
#             arrowprops=dict(arrowstyle="->", color="red"),
#         )
#     if pc13[i] != pc13[i - 1]:  # PC13 电平变化
#         plt.annotate(
#             "",
#             xy=(i, pc13[i]),
#             xytext=(i, pc13[i - 1]),
#             arrowprops=dict(arrowstyle="->", color="blue"),
#         )

# for i, trigger in enumerate(triggers):
#     if trigger == 'PA0' and i > 0:  # PA0 触发
#         plt.annotate('', xy=(i, pa0[i]), xytext=(i, pa0[i-1]),
#                      arrowprops=dict(arrowstyle="->", color='red'))
#     elif trigger == 'PC13' and i > 0:  # PC13 触发
#         plt.annotate('', xy=(i, pc13[i]), xytext=(i, pc13[i-1]),
#                      arrowprops=dict(arrowstyle="->", color='blue'))


# 隐藏上方和右方的边框
ax = plt.gca()  # Get current axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# 隐藏x轴和y轴刻度
plt.xticks([])
plt.yticks([])

# 设置图例
plt.legend(loc="upper left")

# 显示波形图
plt.show()
