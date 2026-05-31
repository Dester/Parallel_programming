import numpy as np

def gen_matrix(size: int, file_path: str, min_val: int = 0, max_val: int = 100) -> None:
    matrix = np.random.randint(min_val, max_val + 1, size=(size, size))
    np.savetxt(file_path, matrix, fmt='%d', delimiter=' ')

    print(f"NumPy матрица {size}x{size} сохранена в '{file_path}'")

gen_matrix(size=200, file_path='input/200A.txt')
gen_matrix(size=200, file_path='input/200B.txt')
gen_matrix(size=400, file_path='input/400A.txt')
gen_matrix(size=400, file_path='input/400B.txt')
gen_matrix(size=800, file_path='input/800A.txt')
gen_matrix(size=800, file_path='input/800B.txt')
gen_matrix(size=1200, file_path='input/1200A.txt')
gen_matrix(size=1200, file_path='input/1200B.txt')
gen_matrix(size=1600, file_path='input/1600A.txt')
gen_matrix(size=1600, file_path='input/1600B.txt')
gen_matrix(size=2000, file_path='input/2000A.txt')
gen_matrix(size=2000, file_path='input/2000B.txt')