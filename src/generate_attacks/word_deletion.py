def word_deletion(text, target_indices, remove_punctuation=False):
    tokens = text.split(' ')
    tokens = [x for i,x in enumerate(tokens) if i not in target_indices]
    edited = ' '.join(tokens)
    return [edited]
