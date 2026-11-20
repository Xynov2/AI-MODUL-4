
import numpy as np

# ==========================================
# PERCOBAAN 1: PERCEPTRON NAND, NOR, DAN XOR
# ==========================================

def step_function(x):
    return 1 if x >= 0 else 0


class Perceptron:
    def __init__(self, n_input, learning_rate=0.1):
        self.weights = np.zeros(n_input)
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return step_function(z)

    def train(self, X, y, epochs=20):
        for epoch in range(epochs):
            total_error = 0

            print(f"\n=== EPOCH {epoch+1} ===")

            # Membuat judul tabel
            print(f"{'Soal':<12}{'Kunci Jawaban':<18}{'Prediksi':<18}{'Error':<8}")
            print("-" * 56)

            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction

                # Menampilkan soal, prediksi dan error
                soal = f"({xi[0]}, {xi[1]})"

                print(f"{soal:<12}{target:<18}{prediction:<18}{error:<8}")

                # Memperbarui bobot dan bias
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_error += abs(error)

            print("-" * 56)
            print(f"Total Error Epoch {epoch+1}: {total_error}")

            if total_error == 0:
                print("Training selesai! Semua prediksi benar.")
                break


# ==========================================
# DATA INPUT DAN TARGET
# ==========================================

# Data input
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Target setiap gerbang
y_nand = np.array([1, 1, 1, 0])
y_nor = np.array([1, 0, 0, 0])
y_xor = np.array([0, 1, 1, 0])


# ==========================================
# PERCOBAAN NAND
# ==========================================

print("\n=== PERCEPTRON NAND ===")

perceptron_nand = Perceptron(n_input=2)
perceptron_nand.train(X, y_nand)

print("\nHasil Akhir NAND:")
for xi in X:
    print(xi, "->", perceptron_nand.predict(xi))


# ==========================================
# PERCOBAAN NOR
# ==========================================

print("\n=== PERCEPTRON NOR ===")

perceptron_nor = Perceptron(n_input=2)
perceptron_nor.train(X, y_nor)

print("\nHasil Akhir NOR:")
for xi in X:
    print(xi, "->", perceptron_nor.predict(xi))


# ==========================================
# PERCOBAAN XOR
# ==========================================

print("\n=== PERCEPTRON XOR ===")

perceptron_xor = Perceptron(n_input=2)
perceptron_xor.train(X, y_xor, epochs=20)

print("\nHasil Akhir XOR:")

hasil_xor = []

for xi in X:
    prediksi = perceptron_xor.predict(xi)
    hasil_xor.append(prediksi)
    print(xi, "->", prediksi)

# Menghitung akurasi XOR
akurasi_xor = np.mean(np.array(hasil_xor) == y_xor) * 100

print("\nTarget XOR  :", y_xor)
print("Prediksi XOR:", hasil_xor)
print(f"Akurasi XOR : {akurasi_xor:.1f}%")

# Menampilkan bobot dan bias akhir XOR
print("\nBobot Akhir XOR:")
print("W1   :", round(perceptron_xor.weights[0], 2))
print("W2   :", round(perceptron_xor.weights[1], 2))
print("Bias :", round(perceptron_xor.bias, 2))
