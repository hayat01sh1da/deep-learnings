import os
from numpy.typing import NDArray
from typing import Any

def eval_seq2seq(model: Any, question: NDArray[Any], correct: NDArray[Any], id_to_char: dict[int, str], verbose: bool = False, is_reverse: bool = False) -> int:
    correct = correct.flatten()
    # Head devider
    start_id = correct[0]
    correct  = correct[1:]
    guess    = model.generate(question, start_id, len(correct))
    # Converrt to string
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct  = ''.join([id_to_char[int(c)] for c in correct])
    guess    = ''.join([id_to_char[int(c)] for c in guess])
    if verbose:
        if is_reverse:
            question = question[::-1]
        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)
        is_windows = os.name == 'nt'
        if correct == guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(f'{mark} {guess}')
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(f'{mark} {guess}')
        print('---')
    return 1 if guess == correct else 0
