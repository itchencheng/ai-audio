import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.arange(40)
    y = np.cos(0.2 * np.pi * x)
	
    y2 = np.fft.fft(y)
	
    print(x)
    print(len(y))
    print(len(y2))
	
    #plt.scatter(x, y, color="red")
    plt.scatter(x, y2, color="green")
	
    plt.show()

if __name__ == "__main__":
    main()
