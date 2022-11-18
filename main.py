import torch
from torch import distributions as pyd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_density(num_samples=1000):
    DTYPE = torch.float32
    torch.manual_seed(-1)
    mv_gauss1 = pyd.MultivariateNormal(torch.zeros(1, dtype=DTYPE), torch.eye(1, dtype=DTYPE)*4.)
    gauss1_samples = mv_gauss1.sample_n(num_samples)

    # optimizing such that 10000 samples can be taken in one shot
    mv_gauss2 = pyd.Normal(loc=0.25*torch.square(gauss1_samples), scale=torch.ones(num_samples, dtype=DTYPE))
    gauss2_samples = mv_gauss2.sample()

    x_samples = torch.hstack([gauss1_samples, gauss2_samples])
    # np_samples = sess.run(x_samples)
    plt.scatter(x_samples[:, 0], x_samples[:, 1], s=10, color='red')
    plt.xlim([-5, 30])
    plt.ylim([-10, 10])
    plt.show()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sample_density()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
