import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def generate_fake_data():
    """Generate data that looks like an example given."""
    xs = np.arange(0, 25, 0.05)
    ys = - 20 * 1./(1 + np.exp(-(xs - 5.)/0.3))
    m = xs > 7.
    ys[m] = -20.*np.exp(-(xs - 7.)[m] / 5.)

    # add noise
    ys += np.random.normal(0, 0.2, xs.size)
    return xs, ys


def main():
    xs, ys = generate_fake_data()

    # smooth out noise
    smoothed = gaussian_filter(ys, 3.)

    # find the point where the signal goes above the background noise
    # level (assumed to be zero here).
    base = 0.
    std = (ys[xs < 3] - base).std()
    m = smoothed < (base - 3. * std)
    x0 = xs[m][0]
    y0 = ys[m][0]

    #plt.plot(xs, ys, '.')
    plt.plot(xs, smoothed, '-')
    plt.plot(x0, y0, 'o')
    plt.show()

if __name__ == '__main__':
    main()

