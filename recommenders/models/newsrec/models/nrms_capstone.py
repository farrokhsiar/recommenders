import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.models.newsrec.models.base_model import BaseModel
from recommenders.models.newsrec.models.layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel_Capstone"]

class NRMSModel_Capstone(BaseModel):
    
    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
    ):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS needs word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
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
        """get input and labels for training from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            torch.tensor(batch_data["clicked_title_batch"], dtype=torch.long),
            torch.tensor(batch_data["candidate_title_batch"], dtype=torch.long),
        ]
        input_label = torch.tensor(batch_data["labels"], dtype=torch.float32)
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """get input of user encoder
        Args:
            batch_data: input batch data from user iterator

        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        return torch.tensor(batch_data["clicked_title_batch"], dtype=torch.long)

    def _get_news_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        return torch.tensor(batch_data["candidate_title_batch"], dtype=torch.long)

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
        hparams = self.hparams
        his_input_title = nn.Input(
            shape=(hparams.his_size, hparams.title_size), dtype=torch.long
        )

        click_title_presents = nn.TimeDistributed(titleencoder)(his_input_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = nn.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NRMS
        Args:
        embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input = nn.Input(
            shape=(hparams.title_size,), dtype=torch.long
        )
        embedded_sequences = embedding_layer(sequences_input)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [embedded_sequences] * 3
        )
        y = nn.Flatten()(y)
        news_repr = nn.Dropout(hparams.dropout)(y)

        model = nn.Model(sequences_input, news_repr, name="news_encoder")
        return model

    def _build_nrms(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams

        # input layer
        clicked_title = nn.Input(
            shape=(hparams.his_size, hparams.title_size), dtype=torch.long
        )
        candidate_title = nn.Input(
            shape=(hparams.title_size,), dtype=torch.long
        )

        # embedding layer
        embedding_layer = nn.Embedding(
            hparams.doc_size, hparams.word_emb_dim, weights=[self.word2vec_embedding], trainable=True
        )

        # news encoder
        newsencoder = self._build_newsencoder(embedding_layer)
        # user encoder
        userencoder = self._build_userencoder(newsencoder)

        # get user representation
        user_present = userencoder(clicked_title)

        # get news representation
        news_present = newsencoder(candidate_title)

        # A^T * H
        att_layer = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)
        att_score = att_layer([news_present, user_present])
        # mask
        mask = nn.Lambda(lambda x: -1e30 * (1 - x))(candidate_title)
        att_score = nn.Add()([att_score, mask])
        # news representation after attention
        news_repr = nn.Lambda(lambda x: nn.softmax(x))(att_score)
        news_output = nn.Dot(axes=1)([news_repr, news_present])

        # multi-task
        click_pred = nn.Dense(1, activation="sigmoid", use_bias=False)(news_output)

        model = nn.Model(
            inputs=[clicked_title, candidate_title],
            outputs=[click_pred],
            name="NRMS",
        )

        scorer = nn.Model(
            inputs=[candidate_title],
            outputs=[news_repr],
            name="NRMS_scorer",
        )

        return model, scorer