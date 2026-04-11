import numpy as np
from numpy.typing import NDArray
from typing import Any
from plot_shim import plt

class CountBasedMethod:
    def __init__(self) -> None:
        pass

    def text_to_word_list(self, text: str) -> list[str]:
        text      = text.lower()
        text      = text.replace(',', ' ,')
        text      = text.replace('.', ' .')
        word_list = text.split(' ')
        return word_list

    def _take_out_query(self, query: str, word_to_id: dict[str, int], co_matrix: NDArray[Any]) -> tuple[dict[str, str], NDArray[Any]] | str:
        if query not in word_to_id:
            return '%s is not found.' % query
        query_id  = word_to_id[query]
        query_vec = co_matrix[query_id]
        return {'query': query}, query_vec

    def _cos_similarity(self, x: NDArray[Any], y: NDArray[Any], eps: float = 1e-8) -> float:
        nx             = x / (np.sqrt(np.sum(x ** 2)) + eps)
        ny             = y / (np.sqrt(np.sum(y ** 2)) + eps)
        cos_similarity = np.dot(nx, ny)
        return cos_similarity

    def _calc_cos_similarity(self, vocab_size: int, co_matrix: NDArray[Any], query_vec: NDArray[Any]) -> NDArray[Any]:
        similarity = np.zeros(vocab_size)
        for i in range(vocab_size):
            similarity[i] = self._cos_similarity(co_matrix[i], query_vec)
        return similarity

    def _output_result_asc(self, similarity: NDArray[Any], query: str, id_to_word: dict[int, str], top: int = 5) -> dict[str, float]:
        count  = 0
        result = {}
        for i in (-1 * similarity).argsort():
            if id_to_word[i] == query:
                continue
            result[id_to_word[i]] = similarity[i]
            count += 1
            if count >= top:
                return result

    def preprocess(self, word_list: list[str]) -> tuple[dict[str, int], dict[int, str], NDArray[Any]]:
        word_to_id = {}
        id_to_word = {}
        for word in word_list:
            if word not in word_to_id:
                new_id             = len(word_to_id)
                word_to_id[word]   = new_id
                id_to_word[new_id] = word
        corpus = np.array([word_to_id[w] for w in word_list])
        return word_to_id, id_to_word, corpus

    def create_co_matrix(self, corpus: NDArray[Any], vocab_size: int, windows_size: int = 1) -> NDArray[Any]:
        corpus_size = len(corpus)
        co_matrix   = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        for index, word_id in enumerate(corpus):
            for i in range(1, windows_size + 1):
                left_index  = index - i
                right_index = index + i
                if left_index >= 0:
                    left_word_id                      = corpus[left_index]
                    co_matrix[word_id, left_word_id] += 1
                if right_index < corpus_size:
                    right_word_id                      = corpus[right_index]
                    co_matrix[word_id, right_word_id] += 1
        return co_matrix

    def rank_similarities(self, query: str, word_to_id: dict[str, int], co_matrix: NDArray[Any], vocab_size: int, id_to_word: dict[int, str], top: int = 5) -> dict[str, Any]:
        query_info, query_vec = self._take_out_query(query, word_to_id, co_matrix)
        similarity            = self._calc_cos_similarity(vocab_size, co_matrix, query_vec)
        result                = self._output_result_asc(similarity, query, id_to_word)
        query_info.update(result)
        return query_info

    # Positive Pointwise Mutual Information
    def ppmi(self, co_matrix: NDArray[Any], verbose: bool = False, eps: float = 1e-8) -> NDArray[Any]:
        M     = np.zeros_like(co_matrix, dtype=np.float32)
        N     = np.sum(co_matrix)
        S     = np.sum(co_matrix, axis = 0)
        total = co_matrix.shape[0] * co_matrix.shape[1]
        count = 0
        for i in range(co_matrix.shape[0]):
            for j in range(co_matrix.shape[1]):
                pmi     = np.log2(co_matrix[i, j] * N / (S[j] * S[i]) + eps)
                M[i, j] = max(0, pmi)
                if verbose:
                    count += 1
                    if count % (total // 100) == 0:
                        print('%.1f%% done' % (100 * count / total))
        return M

    def singular_value_deconposition(self, M: NDArray[Any]) -> NDArray[Any]:
        U, S, V = np.linalg.svd(M)
        return U

    def save_svd_plot_image(self, word_to_id: dict[str, int], U: NDArray[Any], file_path: str) -> None:
        for word, word_id in word_to_id.items():
            plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
        plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
        plt.savefig(file_path)
