from typing import Any, List

def transpose(matrix: List[List[Any]]) -> List[List[Any]]:
    return [list(row) for row in zip(*matrix)]

if __name__ == "__main__":
    # Example usage
    matrix = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    transposed_matrix = transpose(matrix)
    print("Transposed Matrix:", transposed_matrix)

