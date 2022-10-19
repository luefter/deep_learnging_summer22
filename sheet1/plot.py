import matplotlib.pyplot as plt 
import numpy as np 
import pickle 


with open('./sheet1/loss_records_task1.pickle', 'rb') as file:
    loss_records = pickle.load(file)

for t in ["train", "test"]:
    if t == "train":
        number_batches = len(loss_records["SGD"][t])//64
        number_train = number_batches * 64
        plt.plot(np.arange(0, number_batches)*64, np.array(loss_records["SGD"][t][:number_train]).reshape((64,-1)).mean(0), color="r", label="SGD")
        plt.plot(np.arange(0, number_batches)*64, loss_records["Adam"][t][:number_batches], color="b", label="ADAM")
        plt.xlabel('#data')

    else:
        plt.plot(np.arange(0, len(loss_records["SGD"][t])), loss_records["SGD"][t], color="r", label="SGD")
        plt.plot(np.arange(0, len(loss_records["Adam"][t])), loss_records["Adam"][t], color="b", label="ADAM")
        plt.xlabel('#iteration')



    plt.xlabel('#data')
    plt.ylabel('cross entropy loss')
    plt.title(f"loss record - {t} - adam batch size 64")
    plt.legend()
    plt.savefig(f"./sheet1/results/loss_records_optim_{t}.png")
    plt.cla()


# Task 2
with open('./sheet1/loss_records_task2.pickle', 'rb') as file:
    loss_records = pickle.load(file)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('loss record - different batch size')
for batch_size in loss_records.keys():
    print(batch_size)
    #ax1.plot(np.arange(len(loss_records[batch_size]["train"])), loss_records[batch_size]["train"], label= batch_size)
    #ax2.plot(np.arange(len(loss_records[batch_size]["train"])), loss_records[batch_size]["train"], label= batch_size)


    ax1.plot(np.arange(len(loss_records[batch_size]["train"]))*batch_size, loss_records[batch_size]["train"], label=batch_size)
    ax2.plot(np.arange(len(loss_records[batch_size]["test"]))*batch_size, loss_records[batch_size]["test"], label=batch_size)
    print(len(loss_records[batch_size]["test"]))
    #ax1.plot(np.arange(len(loss_records[batch_size]["train"][::int(128/batch_size)]))*128, loss_records[batch_size]["train"][::int(128/batch_size)],label=batch_size)
    #ax2.plot(np.arange(len(loss_records[batch_size]["test"][::int(128/batch_size)])), loss_records[batch_size]["test"][::int(128/batch_size)],label=batch_size)
ax1.legend()
ax2.legend()
fig.savefig("./sheet1/results/loss_records_batch.png")
plt.clf()
