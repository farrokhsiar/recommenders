# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow.keras as keras
from tensorflow.keras import layers


from recommenders.models.newsrec.models.base_model import BaseModel
from recommenders.models.newsrec.models.layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel"]


class NRMSModel(BaseModel):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
    ):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key settings such as head_num and head_dim are there.
            iterator_creator_train (object): NRMS data loader class for train data.
            iterator_creator_test (object): NRMS data loader class for test and validation data
        """
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)

        super().__init__(
            hparams,
            iterator_creator,
            seed=seed,
        )

    def _get_input_label_from_iter(self, batch_data):
        """get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """get input of user encoder
        Args:
            batch_data: input batch data from user iterator

        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        return batch_data["clicked_title_batch"]

    def _get_news_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        return batch_data["candidate_title_batch"]

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """

        # define the input tensor for user's history news click title
        hparams = self.hparams

        # his_input_title is a tensor that represents the user's history of news click titles. 
        # It has a shape of (batch_size, his_size, title_size), where batch_size is the number of users in a batch, his_size is the number of news articles that each user has clicked in the past, and title_size is the number of words in each news article title.
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )

        # pass the user's history news click title through the titleencoder to get the click title presents
        # TimeDistributed layer applies the titleencoder to each element of the his_input_title tensor along the his_size dimension.
        # This means that the titleencoder will be applied his_size times, once for each news article in the user's history.
        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)

        # apply the self-attention mechanism on click title presents
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )

        # apply another attention layer on the output of the self-attention mechanism to get the user present
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams

        # defines a keras input layer for the news article titles
        # hparams.title_size specifies the maximum length of the input sequences.
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")

        # applies the input sequences to the embedding layer to obtain a dense representation for each word in the sequence.
        embedded_sequences_title = embedding_layer(sequences_input_title)

        # This line applies a self-attention mechanism to the input sequences using the SelfAttention layer with hparams.head_num number of heads, each with hparams.head_dim dimensionality. 
        # The self.seed argument sets the random seed for reproducibility.
        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)

        #  apply self-attention mechanism to the input sequences using the SelfAttention layer with hparams.head_num number of heads, each with hparams.head_dim dimensionality.
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)

        # apply a second attention layer, AttLayer2, to the output of the previous dropout layer to obtain a fixed-length representation of the news article titles. 
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams

        #his_size: maximum number of historical titles that will be fed into the model
        #title_size: length of each title (in terms of word embeddings)
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )

        # number of negative samples that will be generated for each positive sample during training
        # total number of titles that will be fed into the model for each training example
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )

        #This input layer is used for making predictions on a single title.
        pred_input_title_one = keras.Input(
            shape=(
                1,
                hparams.title_size,
            ),
            dtype="int32",
        )

        #change the shape of the tensor from (1, hparams.title_size) to (hparams.title_size,),
        #which is the required input shape for the model's output layer.
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(
            pred_input_title_one
        )

        # For his_input_title, an example input tensor might look like [[1, 2, 3], [4, 5, 6], [7, 8, 9]], where each row represents a historical title and each column represents a word embedding for that title.

        # For pred_input_title, an example input tensor might look like [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], where the first row represents a positive sample and the other three rows represent negative samples.

        # For pred_input_title_one, an example input tensor might look like [[1, 2, 3, 4, 5]], where the single row represents a single title.


        # The Embedding layer maps each word in the input text to a vector representation. 
        # self.word2vec_embedding array is a pre-trained word embedding matrix that is used to initialize the weights of this layer.
        # hparams.word_emb_dim specifies the dimensionality of the embedding space
        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        # use the embedding_layer to encode the input text into a fixed-length vector representation
        titleencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder


        # Pass the user's click history, his_input_title, through the userencoder layer to get a fixed-length vector representation of the user's interests.
        user_present = self.userencoder(his_input_title)

        # Pass the candidate news articles, pred_input_title, through the newsencoder layer to get a fixed-length vector representation of each article. 
        # The TimeDistributed wrapper applies the same newsencoder to each time step of the input sequence. 
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)

        # Pass a single candidate news article, pred_title_one_reshape, through the newsencoder layer to get a fixed-length vector representation.
        news_present_one = self.newsencoder(pred_title_one_reshape)


        # Compute the dot product between each candidate news article representation and the user representation to obtain a measure of their similarity. 
        # The axes=-1 argument indicates that the dot product is taken along the last axis of the two inputs, which corresponds to the vector dimension.
        preds = layers.Dot(axes=-1)([news_present, user_present])

        #This line is applying the softmax function to the similarity scores to obtain a probability distribution over the candidate articles.
        preds = layers.Activation(activation="softmax")(preds)


        #This line is computing the dot product between the single candidate news article representation and the user representation to obtain a similarity score.
        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])

        #This line is applying the sigmoid function to the similarity score to obtain a probability that the user will click on the single candidate article.
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([his_input_title, pred_input_title], preds)
        scorer = keras.Model([his_input_title, pred_input_title_one], pred_one)

        return model, scorer
