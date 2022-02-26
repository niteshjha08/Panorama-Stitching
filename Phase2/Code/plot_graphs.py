#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

def plot():
    pass

if __name__=="__main__":
    file = open("/home/nitesh/programming/CMSC733/P1/niteshj_p1/Phase2/Checkpoints2/Logs/errors_history.txt",'r')
    errors=file.readlines()
    # print(type(float(errors[0])))
    print(errors)
    errors=[np.float32(error) for error in errors]
    print(errors)
    plt.plot(errors)
    plt.xlabel('epoch')
    plt.ylabel('mean absolute error')
    plt.show()
    # plot()