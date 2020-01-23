import fasttext

model = fasttext.load_model("./bin/ft_skipgram_model.bin")

def get_similar_word_count(threshold=0.6):
    for i, word in enumerate(model.words):
        yield [word, len([x for x,y in model.get_nearest_neighbors(word, len(model.words)) if y > threshold])]

most_similars_precalc = {word : model.get_nearest_neighbors(word, 10) for word in model.words}
sorted_words = sorted(list(get_similar_word_count()), reverse=True, key=lambda kv: kv[1])

with open("skipgram_res.txt", "w") as file:
    for word, count in sorted_words:
        similar_words = ' '.join([w[0] for w in most_similars_precalc[word]])
        file.write(word + "\t\t" + str(count) + " " + similar_words + "\n")