# The following code implements the Viterbi algorithm for the Part of Speech tagging.
# The training data for the following code has been taken from the site: "https://www.clips.uantwerpen.be/conll2000/chunking/".

import pandas as pd 					# Importing pandas library for using dataframes for storing the training data from 'train.txt' file and performing computations.
import nltk								# Importing nltk to create a bigram list of all the tags present in the training data.

# Opening of the file containing the training data 'train.txt' and extracting its data.
file = open('train.txt', 'r')			# Opening the file 'train.txt' that contains the training data.
data = []								# Creating a list called 'data' that will be a list of lists in which each list item(a list) will contain a line of the file, i.e., a word and its corresponding PoS tag and chunk tag.
for line in file:						# Using a for loop to iterate over the file 'train.txt' through each line of the file so that the file's content can be stored in the list 'data'.
	data.append(line.split())			# Converting each line of the file 'train.txt' to a list and than appending it to the list 'data' as a list item.

# Pre-processing of the training data so that the training data can be used as desired.
df = pd.DataFrame.from_records(data)	# Converting the list 'data' to a dataframe called 'df'.
df = df.drop(df.columns[2], axis=1)		# Removing the last column from the dataframe 'df' that contains chunk tags that are not required for the training data.
df = df[~df[1].isin([None])]			# Removing all the rows from the dataframe 'df' in which all the entries are of 'None' values as they are not part of the training data.
dataset = df.groupby(df.columns.tolist(), as_index = False).size()				# Grouping the rows of the dataframe 'df' by word and its PoS tag along with the count(of occurenece of the word and tag pair) associated with them in a new column and storing this new dataframe as 'dataset'.
dataset.to_csv('Dataset.csv')													# Saving the dataframe 'dataset' as a CSV file named 'Dataset.csv'

# Creating a tag transition bigram matrix containing the count for each bigram so that it can be used for tag transition probabilities.
tags_list = df[1].tolist()														# Extracting the column from the dataframe 'df' containing the PoS tag of each word and converting this	column to a list and storing it in a list called 'tags_list'.
tag_bigram_list = list(nltk.bigrams(tags_list))									# Creating a bigram of all the tags present in the list 'tags_list' by applying 'nltk' extension and storing this new list of bigrams in a list called 'tag_bigram_list'.

# Creating a bigram matrix as a dataframe named as 'tags_bigram_matrix' from the bigram list 'tag_bigram_list'.
words = sorted(list(set([item for t in tag_bigram_list for item in t])))	
tags_bigram_matrix = pd.DataFrame(0, columns=words, index=words)
for i in tag_bigram_list:
	tags_bigram_matrix.at[i[0],i[1]] += 1
tags_bigram_matrix.to_csv('Tags_bigram_matrix.csv')										# Saving the dataframe 'tags_bigram_matrix' as a CSV file named 'Tags_bigram_matrix.csv'.

# Creating some lists that will be used for performing the required calculation throughout the code.
tags_only = df.drop(df.columns[0], axis=1)												# Creating a dataframe 'tags_only' by removing the first column from the dataframe 'df'(that contains the word in the training data), therefore this new dataframe only contains the PoS tags.
tags_count = tags_only.groupby(tags_only.columns.tolist(), as_index=False).size()		# Grouping the rows of the dataframe 'tags_only' by PoS tag along with its count associated with it in another column.
tags_unique = tags_count[1].unique().tolist()											# 'tags_unique' is a list of all the unique tags present in the dataset. This list is made from the dataframe tags_count's first column that contains all the unique tags.
tag_counting = tags_count['size'].tolist()												# 'tags_counting' is a list of counts of each unique tag present in the dataset. This list is made from the dataframe tags_count's second column.
indexx = tag_counting.index(max(tag_counting))											# 'indexx' contains the index of the most occurring tag in the dataset. This tag will be used when a new word is encountered that is not present in the dataset.

data = input("Enter the string on which you would like to apply Part of Speech tagging:-\n")	# Taking string input from the user to apply PoS tagging.

# The input taken from the user needs some pre-processing so that it can be used in the desired form, so that all the ambiguities from the input data are removed by breaking the input into different parts and then finally converting it into a list.
updated_data = ""												# An empty string 'updated_data' is created that will be filled by the string 'data' in the required way as we iterate over the input data using the for loop.						
lent = len(data)												# 'lent' contains the length of the input string 'data' so that it can be used while iterating over each character present in the input string 'data' using for loop.
for i in range(lent):											# 'for' loop is used for iterating over the input data.
	char = data[i]												# 'char' contains the i-th character of the input 'data'.
	if char.isdigit():											# Checking whether the i-th character 'char' is a digit or not. If it's a digit then it is simply appended to 'updated_data'.
		updated_data += char									# Appending 'char' to 'updated_data' if 'char' is a digit.
		continue												# 'continue' keyword so that the code does not run further in the 'for' loop if 'char' is appended to 'updated_data' in case 'char' is a number.
	else:														# If 'char' is not a number, then 'else' part is checked for the i-th character.
		if ord(char.lower()) in range(97, 123):					# If i-th character in input 'data' is an alphabet then it is simply appended to 'updated_data'.
			updated_data += char 								# Appending 'char' to 'updated_data' if 'char' is an alphabet.
		elif char in ( "-", "&", "/", "\\"):					# If 'char' is a special character from any of these stated special characters, then it is simply appended to 'updated_data'. 
			updated_data += char 								# Appending 'char' to 'updated_data' if 'char' is any of the above stated special character.
		elif char == ".":										# If i-th character is a full-stop(.), then there are some conditions on the basis of which it is handled.
			if i == lent-1:										# If full-stop is the last character in the input sentence, then it is appended to 'updated_data' preceded by a space so that if there is no space between the last word and the full-stop, then it is put in between them, so that they are both handled as separate words while applying PoS tagging. 
				updated_data += " " + char 						# Appending a space followed by 'char' to 'updated_data'.
			else:												# If full-stop is not the last character in the input sentence, then it is simply appended to the 'updated_data'.
				updated_data += char 							# Appending 'char' to 'updated_data'.
		elif char == ",":										# If i-th character is a comma(,), then there are some conditions on the basis of which it is handled.
			if data[i-1].isdigit() and data[i+1].isdigit():		# If comma is preceded and followed by a digit, then it is simply appended to 'updated_data'.
				updated_data += char 							# Appending 'char' to 'updated_data'.
			else:												# If comma does not satisfy the above 'if' condition, then it is appended to 'updated_data' being followed and preceded by a space so that in the input if the comma is just preceded and followed by any other character except a digit then it is disconnected with the help of a space input and handled separately while applying PoS tagging.
				updated_data += " " + char + " "				# Appending 'char' to 'updated_data' being followed and preceded by space.
		elif char == "'":										# If i-th character is an apostrophe('), then there are some conditions on the basis of which it is handled.
			if ord(data[i-1].lower()) in range(97, 123) and ( i-2 < 0 or data[i-2] == " "):					# If 'char'(apostrophe) is preceded by a space or starting of the line which is then followed by a character(Like "D'":Starting of the line or " D'":Preceded by a space), then 'char' is simply appended to 'updated_data'.
				updated_data += char 							# Appending 'char' to 'updated_data'.
			elif data[i-1]+data[i]+data[i+1] == "n't" :			# If the characters i-i,i,i+1 make the string "n't", then the last character of the 'updated_data'(which is 'n') is removed and the string " n'", that is preceded by a space is added to 'updated_data'. This is done so that in words like "weren't", n't is handled as a different word while giving its PoS tag.
				updated_data = updated_data[:-1]				# Removing the last character('n') from 'updated_data'.
				updated_data += " n'"							# Appending " n'" to 'updated_data'.
			else:												# If apostrophe does not fall in any of the above conditions, then is is simply appended to 'updated_data' preceded by a space so that apostrophe is handled as a separate word while applying PoS tag.
				updated_data += " " + char 						# Appending 'char' to 'updated_data' preceded by a space.
		else:													# If 'char' is a character other than all the characters specified in the above conditons then 'char' is appended to 'updated_data' preceded and followed by a space so that such characters are handles separately while applying PoS tag.
			updated_data += " " + char + " "					# Appending 'char' to 'updated_data' preceded and followed by a space.
# All the pre-processing of the input 'data' is done and the processed input data is now present in the 'updated_data'.

data_list = updated_data.split()								# The input 'data' which was processed and then taken in the variable 'updated_data' is now converted to a list called 'data_list' so that each word of the input in handled separately and processed for applying its PoS tag.
print("\nThe output for the given input is as given below:-")

# Below is the implementation of the required Viterbi algorithm that applies PoS tags to each input word and prints the output.
prob = [1]*len(tags_unique)										# 'prob' is a list whose length is equal to the length of the list 'tags_unique', thereby representing the probabilities of each tag for each input word. This list is initialized with '1' to represent the initial Viterbi values.
for i in data_list:												# Iterating over the input using for loop in which 'i' represents the current word in the input that is being processed.
	index = dataset.index[dataset[0] == i]						# Getting the index(s) of the word 'i' from the dataframe 'dataset'.
	possible_tags = dataset.iloc[index, [1, 2]]					# From the index(s)(retrieved from the line of code just above), now we are getting all the tags(associated with the input word 'i') and there associated count and storing it in a dataframe named 'possible_tags'.
	tag = possible_tags[1].tolist()								# 'tag' is a list of tags that are possible for the word 'i'.
	val = possible_tags['size'].tolist()						# 'val' is a list containing the count, i.e., the number of times each possible tag has ocurred for the word 'i'.
	lth = len(val)												# 'lth' is equal to the length of the list 'val' or 'tag', as they both are of equal size.

	if data_list.index(i) == 0:									# Checking whether the i-th word is the first word. If it is the first word then it will handled in a different way as comapred to other words in the input.
	 	index = tags_count.index[tags_count[1] == '.']			# Finding the index of the tag '.' in the dataframe 'tags_count', as we are considering this tag(.) to be the starting tag(like <s> for starting of the line) as this tag(.) represents the end of the line and also represents the starting of a new line, i.e., after this tag we can say that the starting word comes.
	 	valt = tags_count['size'][index].tolist()				# 'valt' is a list containing only a single element, i.e., the number of times the tag '.' has come in the dataset so that its count can be used to get probabilities for every tag that can be the starting word's tag.
	 	valt = valt[0]											# Converting valt from 'list' datatype to 'int' datatype as there is only one element in the list 'valt'.
	 	
	 	all_val = tags_bigram_matrix.loc['.']					# Getting all the bigram values from the dataframe 'tags_bigram_matrix' to get the tag transition probability of every tag present in the dataset for the starting word.
	 	j = -1													# The following 'for' loop gets the probability value for each tag to be the starting word's tag by dividing the occurrence of each after the tag (.) and dividing it by the total occurrence of the tag (.).
	 	for p in all_val:
	 		j += 1
	 		prob[j] = p/valt 									# 'prob' contains starting probabilities of every tag after the 'for' loop has run completely.

	 	idx = [0]*lth 											
	 	for j in range(lth):
	 		idx[j] = tags_unique.index(tag[j])					# 'idx' contains the index of all the possible tags from the list 'tags_unique' for the current word 'i'. All the possible tags are present in the list 'tag' that is declared at the starting of the 'for' loop that iterates over the input.

	 	if len(idx) == 0:										# If the word-i is not present in the dataset, then no tag will be found for the word-i. So, if the word-i is a number than the following code checks for it and if it turns out to be a number then 'CD' tag is given to the word-i.  
	 		if i.isdigit():										# Checking whether word-i is a number or not.
	 			print("{", i, "|CD} ", sep="", end=" ")			# If word-i is a number then 'CD' tag is given as its PoS tag.
	 			for j in range(len(prob)):						# The given 'for' loop updates the probability values in the list 'prob' by setting 'prob' values to zero for every tag except the 'CD' tag. It is done so that the 'prob' reflects that only one tag, i.e., 'CD' is possible for the word-i.
	 				if j != tags_unique.index("CD"):
	 					prob[j] = 0
	 			continue										# Since first word's tag has been given, we need to simply jump on to the next word, without running the further code.
	 		else:												# If the word-i is a number but it contains comma or decimal(like the number: 123,456.789) then the following code will remove all the ',' and '.' from the word-i and then check whether word-i is a number or not.
	 			wrd = ""
	 			for ch in i:
	 				if ord(ch) in range(48, 58):
	 					wrd += ch
	 			if wrd.isdigit():
	 				print("{", i, "|CD} ", sep="", end=" ")
	 				for j in range(len(prob)):					# The given 'for' loop updates the probability values in the list 'prob' by setting 'prob' values to zero for every tag except the 'CD' tag. It is done so that the 'prob' reflects that only one tag, i.e., 'CD' is possible for the word-i.
	 					if j != tags_unique.index("CD"):
	 						prob[j] = 0
	 				continue									# Since first word's tag has been given, we need to simply jump on to the next word, without running the further code.

	 		print("{", i, "|", tags_unique[indexx], "} ", sep="", end=" ")				# If the starting word-i is not a number and not even present in the dataset then the most occurring tag is given to such words. 'indexx' contains the index of the most occurring tag from the list tags_unique.
	 		for j in range(len(prob)):							# The given 'for' loop updates the probability values in the list 'prob' by setting 'prob' values to zero for every tag except for the most occurring tag whose index value is 'indexx'. It is done so that the 'prob' reflects that only one tag is possible for the word-i.
	 			if j != indexx:
	 				prob[j] = 0
	 		continue											# Since the first word's tag has been given we need to skip the further code and continue on to the next word and check for its tag.

	 	for j in range(len(prob)):								# Since some possible tags for the starting word have been found, only those tags should have non-zero probability value and other tags' probability values should be zero. And, the given for loop performs just that.
	 		if j not in idx:									# 'idx' is the list of index of those tags that are possible for the word-i.
	 			prob[j] = 0

	else:														# If the word-i is not a starting word then the following is executed.
		prt = [0]*lth
		idr = [0]*lth
		for t in range(lth):															# The given 'for' loop gets the tag-transition probabilities for each possible tag(present in the list 'tag') of the word-i.
			tag_transition =  tags_bigram_matrix[tag[t]].tolist()						# Getting the count for every tag for reaching the required tag present in the list 'tag' so that this count can be used to get tag-transition probability. These values are stored in the list 'tag_transition'.
			tag_transition = [k / j for k, j in zip(tag_transition, tag_counting)]		# Values in tag_transition list are now element-wise divided by the values present in the list 'tag_counting' to get the tag-transition probabilities for each tag.
			prx = [k*j for k, j in zip(tag_transition, prob)]							# Now, these tag-transition probabilities are now multiplied with the already existing probability values present in the list 'prob' to get the final probabilities for each tag. These values are stored in the list 'prx'.
			prt[t] = max(prx)															# Now the probability which is maximum is selected from the list 'prx' for the desired tag and stored in a list called 'prt'.
			idr[t] = tags_unique.index(tag[t])											# Also, the index of the tag(s) that is/are possible for the word-i is stored in the list 'idr'.

		if lth != 0:																	# The below code updates the above computed probability values(present in the list 'prt') to the 'prob' list at the index(s) present in the list 'idr' and all other probability values are set to zero as these tags are not possible for word-i. But this code runs only when word-i is present in the dataset. If the word-i is not present in the dataset, then all the values in the list 'prob' will be set to zero if the following code runs, which is not desired.
			prob = [0]*len(prob)
			for t in range(lth):
				prob[idr[t]] = prt[t]

	if lth == 0:																		# The below code runs only when the word-i is not present in the dataset.
		if i.isdigit():																	# Checking whether the word-i, that is not present in the dataset is a number or not. The same step like this was performed above.
	 		print("{", i, "|CD} ", sep="", end=" ")										# If word-i is a digit then 'CD' tag is given as its PoS tag.
	 		tag_transition =  tags_bigram_matrix["CD"].tolist()							# The following code calculates the tag-transition probability for the only possible tag 'CD' as done above and then updates it in the list 'prob'.
	 		tag_transition = [k / j for k, j in zip(tag_transition, tag_counting)]
	 		prx = [k*j for k, j in zip(tag_transition, prob)]
	 		for j in range(len(prob)):
	 			if j != tags_unique.index("CD"):
	 				prob[j] = 0
	 			else:
	 				prob[j] = max(prx)
	 		continue																	# If the word-i is found to be a number then the 'continue' keyword skips the execution of the further code and takes onto the next word in the input.
		else:																			# If the word-i is not found to be a number then the following code removes any comma and full-stops if present in the word and then checks whether the word-i is a number or not. Like 'i' can be: '123,456.789', containing comma and full-stop.
	 		wrd = ""
	 		for ch in i:
	 			if ord(ch) in range(48, 58):
	 				wrd += ch
	 		if wrd.isdigit():
	 			print("{", i, "|CD} ", sep="", end=" ")
	 			tag_transition =  tags_bigram_matrix["CD"].tolist()						# The following code calculates the tag-transition probability for the only possible tag 'CD' as done above and then updates it in the list 'prob'.
	 			tag_transition = [k / j for k, j in zip(tag_transition, tag_counting)]
	 			prx = [k*j for k, j in zip(tag_transition, prob)]
	 			for j in range(len(prob)):
	 				if j != tags_unique.index("CD"):
	 					prob[j] = 0
	 				else:
	 					prob[j] = max(prx)
	 			continue																# If the word-i is found to be a number then the 'continue' keyword skips the execution of the further code and continues to the next word in the input.

		print("{", i, "|", tags_unique[indexx], "} ", sep="", end=" ")					# If the word-i is not a number and not present in the dataset then the most occurring tag is given to the word-i.
		tag_transition =  tags_bigram_matrix[tags_unique[indexx]].tolist()				# The following code calculates the tag-transition probability for the only possible tag, i.e., the most occurring tag as done above and then updates it in the list 'prob'.
		tag_transition = [k / j for k, j in zip(tag_transition, tag_counting)]
		prx = [k*j for k, j in zip(tag_transition, prob)]
		for j in range(len(prob)):
			if j != indexx:
				prob[j] = 0
			else:
				prob[j] = max(prx)
		continue											# Since the most occurring tag has been given to the word-i there is no need to execute the code further, therefore we continue onto the next word. 

	if lth > 1:												# If there are more than one possible tags for the word-i then the following code is executed.
		idx = [0]*lth
		prb = [0]*lth
		sums = sum(val)										# 'sums' contains the sum of the values present in the list 'val' that can be used to calculate the likelihood probabilities. 'val' is a list containing the count, i.e., the number of times each possible tag has ocurred for the word 'i'.
		for j in range(lth):								# This loop calculates the likelihood probability for each by dividing that tag's occurrence value prsent in 'val' by 'sums'.
			prb[j] = val[j]/sums							# 'prb' contains the likelihood probability of tags.
			idx[j] = tags_unique.index(tag[j])				# 'idx' contains the index of tags.
		for j in range(lth):
			prob[idx[j]] = prb[j]*prob[idx[j]]				# Multiplying likelihood probability with already exisiting probability values.

# If there is only one possible tag for a word then there is no need of multiplying likelihood probability to the transition probability because since there is only one possible tag so likelihood probability will be equal to '1'. Therefore, we can prob already has the final probability.
	ind = prob.index(max(prob))								# 'ind' gets the index value of the highest probability value present in the list 'prob' to get the index of the PoS tag that can be used to get the tag from the list 'tags_unique'.
	TAG = tags_unique[ind]									# Getting the output tag from the list tags_unique with the help of the index 'ind'.
	print("{", i, "|", TAG, "} ", sep="", end=" ")			# Printing the output.