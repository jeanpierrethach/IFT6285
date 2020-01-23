import spacy

# Need to run this line to convert the gensim model into spacy format
# python3 -m spacy init-model en ./models/spacy-min-count-100 --vectors-loc models/gensim-model-min-count-100.txt.gz

def computeNeighboringWords(model, top=10, row_count="all", write=False, threshold=0.5):

	# Load spacy model
	nlp = spacy.load("./models/" + model)

	# Fetch list of words from vocab
	vocab = list(nlp.vocab.strings)

	# Initialize list and variables
	targets = []
	neighbors = []
	neighbors_count = []
	scores = []
	count = 0

	# If all, fetch length of vocab
	if (row_count == "all"):
		row_count = len(vocab)

	# Iterate through vocab
	for token1 in vocab:
		if count < row_count:
			# If valid word token1
			if (nlp(token1).vector_norm > 0):
				# Initialize list
				words = []
				score = []
				# Iterate through every other word
				for token2 in vocab:
					# Not equal to word being compared to
					if (token2 != token1):
						# If valid word token2
						if (nlp(token2).vector_norm > 0):
							# Calculate similarity score between token1 and token2
							sim_score = nlp(token1).similarity(nlp(token2))
							# Append results to list
							words.append(token2)
							score.append(sim_score)
				# If word was valid
				if len(words) > 0:

					# Sort in descending order by similarity scores
					list1, list2 = (list(t) for t in zip(*sorted(zip(score, words), reverse=True)))

					list1 = [x for x in list1 if x > threshold]
					list2 = list2[:len(list1)]

					# Fetch top neighboring words
					targets.append(token1)
					neighbors_count.append(len(list1))
					if len(list1) > top:
						neighbors.append(list2[:10])
						scores.append(list1[:10])
					else:
						neighbors.append(list2)
						scores.append(list1)

					# Increment count by one if not counting all
					count += 1

	output = []
	for i in range(0, len(targets)):
		'''
		print("Find top " + str(top) + " most similar words to " + targets[i] + ":")
		print(neighbors[i])
		print(scores[i])
		'''

		#output.append(targets[i] + "     " + str(neighbors_count[i]) + "     " + " ".join(neighbors[i]))
		output.append('{:>12}  {:>12}  {:>12}'.format(targets[i], str(neighbors_count[i]), " ".join(neighbors[i])))

	with open("output/out-" + model + ".txt", "w") as filehandle:
		filehandle.writelines("%s\n" % line for line in output)

computeNeighboringWords(model="spacy-min-count-100", top=12, row_count=100, threshold=0.42)