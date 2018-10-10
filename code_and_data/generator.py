import numpy as np
from gensim.models import FastText
from random import shuffle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import keras
from unidecode import unidecode

def get_FastText_word_embedding(embeding_dir="~/Documents/word_embeddings/wiki.en/wiki.en"):
    print("Loading FastText embedding")
    embedding = FastText.load_fasttext_format(embeding_dir)
    print("Finished loading FastText Embedding")
    return embedding

class FastTextDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files, word_embedding, n_channels=300,
                 n_classes=1, shuffle=True, sentence_padding=512, batch_size=64):
        'Initialization'
        self.sentence_padding = sentence_padding
        self.stop_words = stopwords.words('english')
        self.batch_size = batch_size
        self.word_embedding = word_embedding
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(files_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if True == self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((len(files_temp), self.sentence_padding, self.n_channels))
        y = np.zeros(len(files_temp), dtype=int)

        # Generate data
        for i, file in enumerate(files_temp):
            # Store sample
            try:
                with open(file, "r") as f:
                    j = 0
                    for line in f.readlines():
                        line = unidecode(line).lower()
                        if j >= self.sentence_padding:
                            break
                        if line.split(": ")[0] in ["message-id", "date", "from", "to", "mime-version",
                                                   "content-type", "content-transfer-encoding", "x-from", "x-cc",
                                                   "x-bcc", "x-folder", "x-origin", "sent", "cc", "received",
                                                   "content-length", "x-mimeole", "x-mailer", "x-apparently-to",
                                                   "x-track"]:  # "x-filename"
                            continue
                        elif line[0:11] == "<markup id=":
                            continue
                        else:
                            words = word_tokenize(line)
                            for word in words:
                                if word.isdigit():
                                    word = "1"
                                if j >= self.sentence_padding:
                                    break
                                elif word not in self.stop_words:
                                    X[i, j, :] = self.word_embedding.wv[word]
                                    j += 1
                y[i] = int("personal" in file)
            except:
                print(file)

        return X, y
