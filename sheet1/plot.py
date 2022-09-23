import matplotlib.pyplot as plt 
import numpy as np 
import pickle 


# with open('loss_records_task1.pickle', 'rb') as file:
#     loss_records = pickle.load(file)


# plt.plot(np.arange(0,len(loss_records["SGD"]["test"])),loss_records["SGD"]["test"],color="r",label="SGD")
# plt.plot(np.arange(0,len(loss_records["Adam"]["test"])),loss_records["Adam"]["test"],color="b",label="ADAM")

# plt.xlabel('#batch')
# plt.ylabel('cross entropy loss')
# plt.title("loss record - batch size 64")
# plt.legend()
# plt.show()


# Task 2
with open('./sheet1/loss_records_task2.pickle', 'rb') as file:
    loss_records = pickle.load(file)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('TITLE')
for batch_size in loss_records.keys():
    print(batch_size)
    ax1.plot(np.arange(len(loss_records[batch_size]["train"]))*batch_size, loss_records[batch_size]["train"],label=batch_size)
    #ax1.plot(np.arange(len(loss_records[batch_size]["train"][::int(128/batch_size)]))*128, loss_records[batch_size]["train"][::int(128/batch_size)],label=batch_size)
    #ax2.plot(np.arange(len(loss_records[batch_size]["test"][::int(128/batch_size)])), loss_records[batch_size]["test"][::int(128/batch_size)],label=batch_size)
    ax2.plot(np.arange(len(loss_records[batch_size]["test"])), loss_records[batch_size]["test"],label=batch_size)
ax1.legend()
ax2.legend()
fig.show()