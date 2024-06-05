average_ID = [3.8, 16.7, 40.9, 11.6, 22.4]
# average_ID = [3.8 , 15.5 , 40.4 , 11.4 , 22.0]
average_OOD = [27.2 , 32.6 , 26.1 , 28.0 , 24.7]
# average_OOD = [21.0 , 27.7 , 20.5 , 26.7 , 19.8]

base_ID = average_ID[0]
base_OOD = average_OOD[0]
def cal_gain(base, average):
    gain = []
    for i in range(len(average)):
        # improve = (average[i] - base) / base  * 100
        improve = (average[i] - base)
        gain.append(improve)
    return gain
ID_gain = cal_gain(base_ID, average_ID)
OOD_gain = cal_gain(base_OOD, average_OOD)
ID_OOD_gain = []
for i in range(len(ID_gain)):
    ID_OOD_gain.append(ID_gain[i] - OOD_gain[i])
# keep 2 decimal
ID_gain = [round(i, 2) for i in ID_gain]
OOD_gain = [round(i, 2) for i in OOD_gain]
ID_OOD_gain = [round(i, 2) for i in ID_OOD_gain]
print(" & ".join([str(i) for i in ID_gain]))
print(" & ".join([str(i) for i in OOD_gain]))
print(" & ".join([str(i) for i in ID_OOD_gain]))