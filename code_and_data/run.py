import tensorflow as tf
import sys
import os
import time
import ujson as json
import urllib.request
import zipfile
from glob import glob
from random import shuffle
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras import optimizers

from generator import get_FastText_word_embedding, FastTextDataGenerator
from model import create_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_version = "v1"
parms = dict(sentence_padding=512,
             word_embedding_size=300,
             lstm1_output_size=50,
             lstm2_output_size=200,
             regularizer=0.0001,
             read_again=False)
batch_size = 128*2
epochs = 20

if __name__ == '__main__':
    fasttext_zip_file = "~/Documents/word_embeddings/wiki.en.zip"
    if not os.path.isfile(fasttext_zip_file):
        os.makedirs("~/Documents/word_embeddings/wiki.en/wiki.en")
        urllib.request.urlretrieve("https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip",
                                   filename=fasttext_zip_file)
        zip_ref = zipfile.ZipFile(fasttext_zip_file, 'r')
        zip_ref.extractall("~/Documents/word_embeddings/wiki.en/wiki.en")
        zip_ref.close()

    sess = tf.Session()
    K.set_session(sess)
    data_folder = glob(os.path.join(os.getcwd(), "data", "*", "*"))
    shuffle(data_folder)
    fast_embedding = get_FastText_word_embedding(embeding_dir="~/Documents/word_embeddings/wiki.en/wiki.en")

    train_set = data_folder[0:int(len(data_folder) * 0.8)]
    val_set = data_folder[int(len(data_folder) * 0.8):int(len(data_folder) * 0.9)]
    test_set = data_folder[int(len(data_folder) * 0.9):]
    data_gen_parms = dict(word_embedding=fast_embedding,
                          n_channels=parms["word_embedding_size"],
                          n_classes=1,
                          shuffle=True,
                          sentence_padding=parms["sentence_padding"],
                          batch_size=batch_size)
    train_gen = FastTextDataGenerator(files=train_set, **data_gen_parms)
    val_gen = FastTextDataGenerator(files=val_set, **data_gen_parms)
    test_gen = FastTextDataGenerator(files=test_set, **data_gen_parms)

    model_time = time.strftime('%y-%m-%d__%H-%M')
    model_path = os.path.join(
        os.path.abspath(os.getcwd()),
        'results',
        'email_classifier',
        model_version,
        model_time)

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    weights_path = os.path.join(model_path, 'weights')
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)
    weights_path = os.path.join(weights_path,
                                'weights.{epoch:02d}-{val_loss:.4f}.hdf5')

    model = create_model(**parms)
    model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(0.001, decay=1e-5), metrics=['accuracy'])
    print('Model created:')
    model.summary()
    with open(os.path.join(model_path, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(os.path.join(model_path, 'parms.json'), 'w') as outfile:
        json.dump(parms, outfile)

    cp = ModelCheckpoint(filepath=weights_path, monitor="val_loss", verbose=1, save_best_only=True)

    csv_log = CSVLogger(os.path.join(model_path,'training.log'))

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=3)

    es = EarlyStopping(monitor="val_loss", patience=4)

    start_train = time.time()
    try:
        model.fit_generator(generator=train_gen,
                            verbose=1,
                            validation_data=val_gen,
                            epochs=epochs,
                            callbacks=[cp, csv_log, reduce_lr, es],
                            workers=4)

        end_train = time.time()
        print("Time spent training: {} sec".format(end_train - start_train))
        model.save(os.path.join(model_path, "final_model.hdf5"))
        print('Evaluating Model...')
        scores = model.evaluate_generator(generator=test_gen,
                                          workers=1)
        met = ['loss'] + model.metrics
        evaluation = []
        for i in range(len(met)):
            if isinstance(met[i], str):
                evaluation.append('{0} : {1}'.format(met[i], scores[i]))
            else:
                evaluation.append('{0} : {1}'.format(met[i].__name__, scores[i]))
        with open(os.path.join(model_path, 'eval-{0}.txt'.format(model_version)), 'w') as f:
            f.write('\n'.join(evaluation))
        print('\n'.join(evaluation))

    except KeyboardInterrupt:
        print("\nTraining interrupted, cleaning up, Ctrl + C to exit")
        end_train = time.time()
        print("Time spent training: {} sec".format(end_train - start_train))
        sys.exit()
