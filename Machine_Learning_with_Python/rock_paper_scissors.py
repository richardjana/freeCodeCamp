from typing import List

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

N = 10

model = Sequential([LSTM(32, input_shape=(N*2, 3)),
                    Dense(3, activation='softmax')
                    ])
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy')

# Helper: map moves
move_to_idx = {'R': 0, 'P': 1, 'S': 2}
idx_to_move = ['R', 'P', 'S']
encoder = OneHotEncoder(sparse_output=False).fit(np.array([[0], [1], [2]]))
beating_move = {'R': 'P', 'P': 'S', 'S': 'R'}  # move to beat the prediction

def player(prev_play: str, opponent_history: List[str] = [], my_history: List[str] = []) -> str:
    """ Player function to be called by the game runner.
    Args:
        prev_play (str): Opponents previous move.
        opponent_history (List[str], optional): History of opponents moves. Defaults to [].
        my_history (List[str], optional): History of my moves. Defaults to [].
    Returns:
        str: My next move.
    """
    # impute starting sequence
    if prev_play == '':
        prev_play = 'R'
        opponent_history.extend(['P']*N)
        my_history.extend(['S']*N)
        my_history.append('P')

        # reset from previous opponent
        while len(opponent_history) > 10:
            del opponent_history[0]
        while len(my_history) > 11:
            del my_history[0]

    # encode history (last entry is my last move -> don't use here)
    hist_idx = np.array([[move_to_idx[m]] for m in (opponent_history+my_history)[:-1]])
    hist_onehot = encoder.transform(hist_idx).reshape(1, 2*N, 3)

    # update model
    target_idx = move_to_idx[prev_play]
    target_onehot = encoder.transform([[target_idx]])
    model.fit(hist_onehot, target_onehot, epochs=1, verbose=0)

    # update history
    opponent_history.append(prev_play)
    del opponent_history[0]
    del my_history[0]

    # predict opponents next move
    hist_idx = np.array([[move_to_idx[m]] for m in (opponent_history+my_history)])
    hist_onehot = encoder.transform(hist_idx).reshape(1, 2*N, 3)

    pred = model.predict(hist_onehot, verbose=0)
    predicted_move = idx_to_move[np.argmax(pred)]

    my_history.append(beating_move[predicted_move])

    return beating_move[predicted_move]
