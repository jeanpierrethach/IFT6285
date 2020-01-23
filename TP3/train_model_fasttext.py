import fasttext

# model='cbow' or 'skipgram'
# dim : dimensionality of the word vector embedding
# min_count : minimal number of word occurences
model = fasttext.train_unsupervised("train_posts.csv", model='cbow', min_count=100)
model.save_model("./bin/ft_cbow_model-mincount100.bin")

# example to get the nearest neighbors based on similarity scores
#print([(x,y) for x,y in model.get_nearest_neighbors("water", 10) if x > 0.5])