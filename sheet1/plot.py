import matplotlib.pyplot as plt 
import numpy as np 
import pickle 


with open('loss_records_task1.pickle', 'rb') as file:
    loss_records = pickle.load(file)


plt.plot(np.arange(0,len(loss_records["SGD"]["test"])),loss_records["SGD"]["test"],color="r",label="SGD")
plt.plot(np.arange(0,len(loss_records["Adam"]["test"])),loss_records["Adam"]["test"],color="b",label="ADAM")

plt.xlabel('#batch')
plt.ylabel('cross entropy loss')
plt.title("loss record - batch size 64")
plt.legend()
plt.show()


