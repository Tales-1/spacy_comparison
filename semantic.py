import spacy

nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# I noticed that cat and monkey have a similarity of 0.59 which makes sense as they are both animals
# The relationship of banana to monkey returned a similarity of 0.4 which shows that the program identified
# some sort of relationship between a banana and monkey which is that a monkey usually eats bananas.
# However when looking at the similarity of a cat to a banana the program returned a similarity of 0.2 
# which is probably due to the fact that cats do not customarily eat bananas which the program has identified correctly

# My own example of a scenario like this
_word1 = nlp("tennis")
_word2 = nlp("football")
_word3 = nlp("racquet")

# Tennis and football return a similarity of 0.54
print(_word1.similarity(_word2))
# Strangely, football and racquet return a similarity of the exact number
print(_word2.similarity(_word3))
# Tennis and Racquet return a similarity of 0.99 which is expected
# However if if the first letter of racquet is capitalised it returns a lower similarity score which is interesting
print(_word3.similarity(_word1))


tokens = nlp('cat apple monkey banana')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)
