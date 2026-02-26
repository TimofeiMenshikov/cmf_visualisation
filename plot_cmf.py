
import matplotlib.pyplot as plt

FILENAME = "data.txt"

LAMBDA_INDEX = 0
I_R_INDEX    = 1
I_G_INDEX    = 2
I_B_INDEX    = 3
I_m_INDEX    = 4

LAMBDA_RED   = 620 # надо потом вынести в конфиг так как копипаст из gamut.py
LAMBDA_GREEN = 530
LAMBDA_BLUE  = 470


def read_data_from_file(filename):
    with open(filename, "r") as file:
        data_raw = file.readlines()

        data = []

        for i in range(len(data_raw)):
            data.append(list(map(float, data_raw[i].split())))


    return data


def plot_cmf(data):


    # Сортируем по длине волны
    data = sorted(data, key=lambda x: x[0])

    R = []
    G = []
    B = []

    for i in range(len(data)):

        lam = data[i][LAMBDA_INDEX]


        if lam <= LAMBDA_BLUE or lam >= LAMBDA_RED:

            R.append(data[i][I_R_INDEX])
            G.append(-data[i][I_G_INDEX])
            B.append(data[i][I_B_INDEX])

        elif lam <= LAMBDA_GREEN:

            R.append(-data[i][I_R_INDEX])
            G.append(data[i][I_G_INDEX])
            B.append(data[i][I_B_INDEX])           

        elif lam <= LAMBDA_RED:
            
            R.append(data[i][I_R_INDEX])
            G.append(data[i][I_G_INDEX])
            B.append(-data[i][I_B_INDEX])  


    wavelength = [row[0] for row in data]


    plt.plot(wavelength, R)
    plt.plot(wavelength, G)
    plt.plot(wavelength, B)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("RGB Color Matching Functions")
    plt.legend(["R", "G", "B"])

    plt.show()


if __name__ == "__main__":

    data = read_data_from_file(FILENAME)
    print(data)

    plot_cmf(data)