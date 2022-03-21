
def word_attribution(tokens, token_attr_scores):
    '''
    return:
    list of tuples where each tuple is of the format (word_level_score, index_list, word).
    word_level_score:   Score of the word calculated by adding scores of all the tokens.
    index_list      :   List of indexes of the tokens that form a word.
    word            :   Complete word created by appending tokens.
    '''
    # Convert tokens to words by joining and calc. score by adding scores of all the tokens in a word
    score_word_list = []
    curr_list = []
    curr_score = 0
    curr_str = ""
    for i in range(len(tokens)):
        if (tokens[i][0] == 'Ġ' and len(curr_list) != 0):
            score_word_list.append((curr_score, curr_list, curr_str))
            curr_score = 0
            curr_list = []
            curr_str = ""

        curr_score += token_attr_scores[i]
        curr_list.append(i)
        curr_str += tokens[i]

    score_word_list.append((curr_score, curr_list, curr_str))
    return score_word_list
