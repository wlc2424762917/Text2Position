import sng_parser
import spacy
# graph = sng_parser.parse('I am located on the west of a gray-colored road.')
# from pprint import pprint
# pprint(graph)
from spacy import displacy

sentence = 'The position is on the west of a gray-colored road.'
nlp = spacy.load('en_core_web_sm')
doc = nlp(sentence)
for entity in doc.noun_chunks:
    print(entity.text, entity.root.dep_)


root = [token for token in doc if token.head == token][0]
print(f"Root: {root}")

# 递归函数遍历所有节点，并显示层数
def traverse_tree(node, level=0):
    print(f"{' ' * level * 2}- {node.text} (Level {level})")
    print(node.pos_, node.tag_,)
    for child in node.children:
        traverse_tree(child, level + 1)

# 从根节点开始遍历
traverse_tree(root)