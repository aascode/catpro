from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

import config

config_params = config.config

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]

def doc2model(documents=None, output_dir='./'):

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    model = Doc2Vec(documents, vector_size=100, window=3, min_count=3, workers=4,epochs=50)

    model.save(output_dir+"my_doc2vec_model")
    return model



def model2vec(input_dir = './', model=None, documents=None):
    model = Doc2Vec.load(input_dir+ "my_doc2vec_model")  # you can continue training with the loaded model!
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    vectors = []
    for document in documents:
        vector = model.infer_vector(document)
        vectors.append(vector)
    vectors = np.array(vectors)
    return vectors




