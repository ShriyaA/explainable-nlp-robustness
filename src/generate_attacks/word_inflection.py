from textattack.transformations import WordSwapInflections
from textattack.shared.attacked_text import AttackedText

def word_inflection(sample, indices_to_swap):
    transformation = WordSwapInflections()
    result = transformation._get_transformations(sample, indices_to_swap)
    result = [x.text for x in result]
    return result


if __name__=='__main__':
    print(word_inflection("The movie is great", [1]))