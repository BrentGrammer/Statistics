# Natural Language Processing

[See chapter 10 of FasAI book](https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb)

- Form of self supervised learning
- a language model predicts the next word of a document
  - A classifier, however, predicts some external label—in the case of IMDb, it's the sentiment of a document.
- Classification is a good area where NLP can solve real world problems. Ex: classifying emals: send this email to HR, send to Payroll dept., is spam, etc.

### Training

- you get even better results if you fine-tune the (sequence-based) language model prior to fine-tuning the classification model.
- the Universal Language Model Fine-tuning (ULMFit) approach, see [paper](https://arxiv.org/abs/1801.06146)
  - this extra stage of fine-tuning of the language model, prior to transfer learning to a classification task, resulted in significantly better predictions.
- When dealing with large documents (over 2000 words), ULMFit might be better - transformers has to read all the document at once whereas ULMFit can read it in pieces and train gradually

#### HuggingFace models

- There are tons of huggingface models at https://huggingface.co/models
- You can probably find a pre-trained model on all sorts of data trained on documents to start with by searching their models and using it.
- microsoft/deberta-v3 is a general purpose model that is good for general use for NLP.

### Transformers

- New architecture for NLP (used instead of RNN recurrent nueral network architecture). Good for taking advantage of optimization accelerators like new TPUs (Tensor Processing Units)
- The Transformers masked model approach is now more popular than the ULMFit model (still same basic idea)
  - The ULMFit basic idea was to start with a tranining set, fine tune it with a specific corpus and then create a classifier

### Steps for NLP modeling

- Tokenization:: Convert the text into a list of words (or characters, or substrings, depending on the granularity of your model)
- Numericalization:: Make a list of all of the unique words that appear (the vocab), and convert each word into a number, by looking up its index in the vocab
- Language model data loader creation:: fastai provides an LMDataLoader class which automatically handles creating a dependent variable that is offset from the independent variable by one token. It also handles some important details, such as how to shuffle the training data in such a way that the dependent and independent variables maintain their structure as required
- Language model creation:: We need a special kind of model that does something we haven't seen before: handles input lists which could be arbitrarily big or small. There are a number of ways to do this; in this chapter we will be using a recurrent neural network (RNN). We will get to the details of these RNNs in the <<chapter_nlp_dive>>, but for now, you can think of it as just another deep neural network.

### Tokenization

3 main ways to tokenize:

- Word-based:: Split a sentence on spaces, as well as applying language-specific rules to try to separate parts of meaning even when there are no spaces (such as turning "don't" into "do n't"). Generally, punctuation marks are also split into separate tokens.
- Subword based:: Split words into smaller parts, based on the most commonly occurring substrings. For instance, "occasion" might be tokenized as "o c ca sion."
- Character-based:: Split a sentence into its individual characters.
- spaCy is a popular python word tokenizer library

- Can use transformers library to tokenize and work with NLP. See [video](https://www.youtube.com/watch?v=toUgBQv1BT8) at around 32:00 timestamp

#### Independent vs. dependent variable

- the independent variable is the text up to the last token in the text
- The dependent variable is the same thing offset by one token: the second token in the set up to the last token.

#### Special tokens

- some tokens that start with the characters "xx", which is not a common word prefix in English. These are special tokens.
  - For example, the first item in the list, xxbos, is a special token that indicates the start of a new text ("BOS" is a standard NLP acronym that means "beginning of stream"). By recognizing this start token, the model will be able to learn it needs to "forget" what was said previously and focus on upcoming words.
  - note that library like spaCy does not add these.
- example tokenization: `['xxbos','xxmaj','this','movie',',','which','i','just','discovered','at','the','video','store',',','has','apparently','sit','around','for','a','couple','of','years','without','a','distributor','.','xxmaj','it',"'s",'easy'...]`
- For instance, the rules will replace a sequence of four exclamation points with a special repeated character token, followed by the number four, and then a single exclamation point. In this way, the model's embedding matrix can encode information about general concepts such as repeated punctuation rather than requiring a separate token for every number of repetitions of every punctuation mark. a capitalized word will be replaced with a special capitalization token, followed by the lowercase version of the word. This way, the embedding matrix only needs the lowercase versions of the words, saving compute and memory resources, but can still learn the concept of capitalization.

Here are some of the main special tokens you'll see:

xxbos:: Indicates the beginning of a text (here, a review)
xxmaj:: Indicates the next word begins with a capital (since we lowercased everything)
xxunk:: Indicates the word is unknown

### Sub word tokenization

- More popular since it can handle different languages (that have combination words without spaces like German)
- Can also handle things like Midi music notation and genomic sequences
- Split tokens into most common recurring sequences of letters
- Tradeoffs: Picking a subword vocab size represents a compromise: a larger vocab means fewer tokens per sentence, which means faster training, less memory, and less state for the model to remember; but on the downside, it means larger embedding matrices, which require more data to learn.

### Numericalization

- map tokens to integers
  - Make a list of all possible levels of that categorical variable (the words are the categories in the vocab).
  - Replace each level with its index in the vocab.
  - note: words that are rare can be replaced with a xxunk special char for unknown for example since there is not enough them for training and it reduces the training set size
- example: `tensor([  2,   8,  21,  28,  11,  90,  18,  59,   0,  45,   9, 351, 499,  11,  72, 533, 584, 146,  29,  12])`

### Fine tuning and loss

- The loss function used by default is cross-entropy loss, since we essentially have a classification problem (the different categories being the words in our vocab). The perplexity metric used here is often used in NLP for language models: it is the exponential of the loss (i.e., torch.exp(cross_entropy))
- Encoder: The model not including the task-specific final layer(s). This term means much the same thing as body when applied to vision CNNs, but "encoder" tends to be more used for NLP and generative models.

### Advancements

- There have been some advancements in improving accuracy by training forwards and backwards text in the model. Experimentation is ongoing and also training in two languages and back to the original has proved to help some as well. More experimentation is ongoing and could be worth exploring to get the accuracy higher than current levels (~95%)
