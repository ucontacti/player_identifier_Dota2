import matplotlib.pyplot as plt
import numpy as np

result_dict = {}
result_dict["accuracy"] = [90.98, 48.85, 78.94, 86.82, 98.58, 79.42, 82.6, 86.89]
result_dict["precision"] = [59.62, 22.93, 27.44, 56.39, 92.58, 20.41, 13.51, 44.77]
result_dict["recall"] = [66.07, 84.74, 17.92, 64.23, 95.69, 35.71, 26.83, 63.35]
result_dict["f1"] = [60.96, 33.69, 13.28, 54.08, 93.79, 22.35, 17.69, 48.19]
result_dict["roc"] = [60.96, 33.69, 13.28, 54.08, 93.79, 22.35, 17.69, 48.19]
result_dict["eer"] = [60.96, 33.69, 13.28, 54.08, 93.79, 22.35, 17.69, 48.19]
# result_dict["eer"] = [0.2122950259787451, 0.4061842945937143, 0.4748245061043829, 0.27589582479164615, 0.04447446223634764, 0.4123521897166277, 0.42668033162134533, 0.2774154231808194]
# arr = np.array(result_dict["eer"])
# q25, q75 = np.percentile(arr, [25, 75])
# bin_width = 2 * (q75 - q25) * len(arr) ** (-1/3)
# bins = round((arr.max() - arr.min()) / bin_width)
# plt.hist(arr, density=True, bins=bins)
# plt.ylabel('Probability')
# plt.xlabel('Data')

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

axs[0].hist(result_dict["accuracy"], density=True)
axs[0].set_title('Accuracy')
axs[0].axis(xmin=0,xmax=100)
axs[1].hist(result_dict["precision"], density=True)
axs[1].set_title('Precision')
axs[1].axis(xmin=0,xmax=100)
axs[2].hist(result_dict["recall"], density=True)
axs[2].set_title('Recall')
axs[2].axis(xmin=0,xmax=100)
plt.ylabel('Probability')
plt.savefig("plots/test_2.png")
fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].hist(result_dict["f1"], density=True)
axs[0].set_title('F1')
axs[0].axis(xmin=0,xmax=100)
axs[1].hist(result_dict["roc"], density=True)
axs[1].set_title('roc')
axs[1].axis(xmin=0,xmax=100)
axs[2].hist(result_dict["eer"], density=True)
axs[2].set_title('eer')
axs[2].axis(xmin=0,xmax=100)
plt.ylabel('Probability')
plt.savefig("plots/test_3.png")
