import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from copy import deepcopy

import random
import numpy as np
import os
import glob

from module import EncoderRNN
from module import DecoderRNN
from text_token import _UNK, _PAD, _BOS, _EOS
from utils import print_time_info, get_time
from utils import check_dir, check_usage, print_curriculum_status
from utils import BLEU, ROUGE
from model_utils import collate_fn, build_optimizer
from logger import Logger

use_cuda = torch.cuda.is_available()
# I think that it's better for us to pack the whole model as a class


class NLG:
    def __init__(
            self,
            batch_size,
            en_optimizer,
            de_optimizer,
            en_learning_rate,
            de_learning_rate,
            attn_method,
            train_data_engine,
            test_data_engine,
            use_embedding,
            en_use_attr_init_state,
            en_hidden_size=100,
            de_hidden_size=100,
            en_vocab_size=None,
            de_vocab_size=None,
            vocab_size=None,
            en_embedding_dim=None,
            de_embedding_dim=None,
            embedding_dim=None,
            embeddings=None,
            en_embedding=True,
            share_embedding=True,
            n_decoders=2,
            cell="GRU",
            n_en_layers=1,
            n_de_layers=1,
            bidirectional=False,
            feed_last=False,
            repeat_input=False,
            batch_norm=False,
            model_dir="./model",
            log_dir="./log",
            is_load=True,
            check_mem_usage_batches=0,
            replace_model=True,
            finetune_embedding=False,
            model_config=None
    ):

        # Initialize attributes
        self.data_engine = train_data_engine
        self.check_mem_usage_batches = check_mem_usage_batches
        self.n_decoders = n_decoders
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.en_embedding_dim = en_embedding_dim
        self.de_embedding_dim = de_embedding_dim
        self.embedding_dim = embedding_dim
        self.repeat_input = repeat_input

        # Initialize embeddings, encoders and decoders
        """
        There are some available options here, most of which matter when using
        E2E dataset.
        (You still can use them while using dialogue generation dataset
        like CMDC, but it's NOT RECOMMENDED.)

        1) en_embedding (default True):
            If the option is on, we're going to add embedding layer into
            encoder; otherwise, the one-hot vectors are directly fed into
            encoder's RNN.
            For now, the decoder always has an embedding layer; this is
            because that we assumed that the decoder should always output the
            natural language, and it's reasonable that using an embedding layer
            instead of directly pass one-hot vectors into RNN.
        2) share_embedding (default True):
            If the option is on, first you should make sure that the input of
            encoder and decoder are in same vector space,
            (e.g. both natural language); otherwise, it will cause some strange
            result, (it is possible that you can train the model without any
            error, but the shared embedding layer doesn't make sense, as you
            should know.)
            When the option is on, the embedding dimension will be the argument
            embedding_dim, and the vocabulary size will be vocab_size; the
            argument en_embedding_dim, de_embedding_dim, en_vocab_size and
            de_vocab_size won't be used.
        3) use_embedding (default True):
            When the option is on:
            (1) If share_embedding option is on, the shared embedding will be
            initialized with the embeddings we pass into the model.
            (2) If en_embedding is on while share_embedding option being off,
            only the embedding in decoder will be initialized with the
            pre-trained embeddings, and the encoder embeddings will be trained
            from scratch (this combination of options is NOT APPROPRIATE when
            using dialogue generation dataset, as you should know, it's kind
            of strange that we only initialize the embedding in decoder when
            both input and output of the encoder and decoder are in same vector
            space.)

        As mentioned above, since that the options are not disjoint, I'll list
        some possible combination below, which are reasonable to be tested and
        compared:

        1) en_embedding=True, share_embedding=True, \
                use_embedding=True (dialogue generation)
        2) en_embedding=True, share_embedding=True, \
                use_embedding=False (dialogue generation)
        3) en_embedding=True, share_embedding=False, \
                use_embedding=True (semantic form to NL)
        4) en_embedding=False, share_embedding=X(don't care), \
                use_embedding=True (semantic form to NL)
        5) en_embedding=True, share_embedding=False, \
                use_embedding=False (semantic form to NL)
        6) en_embedding=False, share_embedding=X(don't care), \
                use_embedding=False (semantic form to NL)

        """
        # embedding layer setting
        if not en_embedding:
            en_embed = None
            de_embed = nn.Embedding(de_vocab_size, de_embedding_dim)
            if use_embedding:
                de_embed.weight = embeddings
                if not finetune_embedding:
                    de_embed.weight.requires_grad = False
        else:
            if share_embedding:
                embed = nn.Embedding(vocab_size, embedding_dim)
                if use_embedding:
                    embed.weight = embeddings
                    if not finetune_embedding:
                        embed.weight.requires_grad = False
                en_embed = embed
                de_embed = embed
            else:
                en_embed = nn.Embedding(en_vocab_size, en_embedding_dim)
                de_embed = nn.Embedding(de_vocab_size, de_embedding_dim)
                if use_embedding:
                    # in E2ENLG dataset, only decoder use word embedding
                    de_embed.weight = embeddings
                    if not finetune_embedding:
                        de_embed.weight.requires_grad = False

        self.encoder = EncoderRNN(
                en_embedding=en_embedding,
                embedding=en_embed,
                en_vocab_size=en_vocab_size,
                en_embedding_dim=(
                    embedding_dim
                    if share_embedding and en_embedding
                    else en_embedding_dim),
                hidden_size=en_hidden_size,
                n_layers=n_en_layers,
                bidirectional=bidirectional,
                cell=cell)

        self.cell = cell
        self.decoders = []
        for n in range(n_decoders):
            decoder = DecoderRNN(
                    embedding=de_embed,
                    de_vocab_size=de_vocab_size,
                    de_embedding_dim=(
                        embedding_dim
                        if share_embedding and en_embedding
                        else self.de_embedding_dim),
                    en_hidden_size=en_hidden_size,
                    de_hidden_size=de_hidden_size,
                    n_en_layers=n_en_layers,
                    n_de_layers=n_de_layers,
                    bidirectional=bidirectional,
                    feed_last=(True
                               if feed_last and n > 0
                               else False),
                    batch_norm=batch_norm,
                    attn_method=attn_method,
                    cell=cell)
            self.decoders.append(decoder)

        self.encoder = self.encoder.cuda() if use_cuda else self.encoder
        self.decoders = [
                decoder.cuda()
                if use_cuda else decoder for decoder in self.decoders]

        # Initialize data loaders and optimizers
        self.train_data_loader = DataLoader(
                train_data_engine,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn,
                pin_memory=True)

        self.test_data_loader = DataLoader(
                test_data_engine,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn,
                pin_memory=True)

        # encoder parameters optimization
        self.encoder_parameters = filter(
                lambda p: p.requires_grad, self.encoder.parameters())
        self.encoder_optimizer = build_optimizer(
                en_optimizer, self.encoder_parameters,
                en_learning_rate)
        # decoder parameters optimization
        decoder_parameters = []
        for decoder in self.decoders:
            decoder_parameters.extend(list(decoder.parameters()))
        self.decoder_parameters = filter(
                lambda p: p.requires_grad, decoder_parameters)
        self.decoder_optimizer = build_optimizer(
                de_optimizer, self.decoder_parameters,
                de_learning_rate)

        print_time_info("Model create complete")
        # check directory and model existence
        Y, M, D, h, m, s = get_time()
        if not replace_model:
            self.model_dir = os.path.join(
                    self.model_dir,
                    "{}{:0>2}{:0>2}_{:0>2}{:0>2}{:0>2}".format(
                        Y, M, D, h, m, s))

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            if not is_load:
                check_dir(self.model_dir)

        self.log_dir = os.path.join(
                self.log_dir,
                "{}{:0>2}{:0>2}_{:0>2}{:0>2}{:0>2}".format(
                    Y, M, D, h, m, s))

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
            os.makedirs(os.path.join(self.log_dir, "validation"))

        with open(os.path.join(self.log_dir, "model_config"), "w+") as f:
            for arg in vars(model_config):
                f.write("{}: {}\n".format(
                    arg, str(getattr(model_config, arg))))
            f.close()

        if is_load:
            self.load_model(model_dir)

        # Initialize the log files
        self.logger = Logger(self.log_dir)
        self.train_log_path = os.path.join(self.log_dir, "train_log.csv")
        self.valid_batch_log_path = os.path.join(
                self.log_dir, "valid_batch_log.csv")
        self.valid_epoch_log_path = os.path.join(
                self.log_dir, "valid_epoch_log.csv")

        with open(self.train_log_path, 'w') as file:
            file.write("epoch, batch, loss, avg-bleu, avg-rouge(1,2,L,BE)\n")
        with open(self.valid_batch_log_path, 'w') as file:
            file.write("epoch, batch, loss, avg-bleu, avg-rouge(1,2,L,BE)\n")
        with open(self.valid_epoch_log_path, 'w') as file:
            file.write("epoch, loss, avg-bleu, avg-rouge(1,2,L,BE)\n")

        # Initialize batch count
        self.batches = 0

        self.en_use_attr_init_state = en_use_attr_init_state

    def train(self, epochs, batch_size, criterion, verbose_epochs=1,
              verbose_batches=1, valid_epochs=1, valid_batches=1000,
              save_epochs=10,
              split_teacher_forcing=False,
              teacher_forcing_ratio=0.5,
              inner_teacher_forcing_ratio=0.5,
              inter_teacher_forcing_ratio=0.5,
              tf_decay_rate=0.9,
              inner_tf_decay_rate=0.9,
              inter_tf_decay_rate=0.9,
              schedule_sampling=False,
              inner_schedule_sampling=False,
              inter_schedule_sampling=False,
              max_norm=0.25, curriculum_layers=2):

        self.batches = 0
        print_curriculum_status(curriculum_layers)

        _teacher_forcing_ratio = teacher_forcing_ratio
        _inner_teacher_forcing_ratio = inner_teacher_forcing_ratio
        _inter_teacher_forcing_ratio = inter_teacher_forcing_ratio

        for idx in range(1, epochs+1):
            epoch_loss = 0
            epoch_BLEU = 0
            epoch_ROUGE = np.array([0, 0, 0, 0])

            # training
            for b_idx, batch in enumerate(self.train_data_loader):
                self.batches += 1
                batch_loss, batch_BLEU, batch_ROUGE = self.run_batch(
                        batch,
                        criterion,
                        curriculum_layers,
                        testing=False,
                        split_teacher_forcing=split_teacher_forcing,
                        teacher_forcing_ratio=_teacher_forcing_ratio,
                        inner_teacher_forcing_ratio=(
                            _inner_teacher_forcing_ratio),
                        inter_teacher_forcing_ratio=(
                            _inter_teacher_forcing_ratio),
                        schedule_sampling=schedule_sampling,
                        inner_schedule_sampling=inner_schedule_sampling,
                        inter_schedule_sampling=inter_schedule_sampling,
                        max_norm=max_norm
                )
                with open(self.train_log_path, 'a') as file:
                    file.write("{}, {}, {}, {}, {}\n"
                               .format(
                                   idx, b_idx+1, batch_loss,
                                   batch_BLEU,
                                   ', '.join(map(str, batch_ROUGE))))
                # tensorbaord logging
                if self.logger:
                    self.logger.write_train(
                            batch_loss, batch_BLEU, batch_ROUGE, self.batches)

                if (self.check_mem_usage_batches
                        and b_idx % self.check_mem_usage_batches == 0):
                    check_usage()
                if self.batches % verbose_batches == 0:
                    print_time_info(
                            "Epoch {} batch {} (batch {}): \
                                    [TRAIN] loss = {}, \
                                    BLEU = {}, ROUGE_(1,2,L,BE) = {}"
                            .format(
                                    idx, b_idx+1, self.batches, batch_loss,
                                    batch_BLEU,
                                    ', '.join(map(str, batch_ROUGE))))

                epoch_loss = (
                        epoch_loss * (b_idx / (b_idx+1))
                        + batch_loss * (1 / (b_idx+1)))
                epoch_BLEU = (
                        epoch_BLEU * (b_idx / (b_idx+1))
                        + batch_BLEU * (1 / (b_idx+1)))
                epoch_ROUGE = (
                        epoch_ROUGE * (b_idx / (b_idx+1))
                        + batch_ROUGE * (1 / (b_idx+1)))

                # validation
                if self.batches % valid_batches == 0:
                    batch = next(iter(self.test_data_loader))
                    valid_loss, valid_BLEU, valid_ROUGE = self.run_batch(
                            batch,
                            criterion,
                            curriculum_layers,
                            testing=True,
                            result_path=os.path.join(
                                os.path.join(self.log_dir, "validation"),
                                "curri_layer{}_epoch{}_batch{}_result.txt"
                                .format(curriculum_layers, idx, b_idx+1)
                            )
                    )
                    with open(self.valid_batch_log_path, 'a') as file:
                        file.write("{}, {}, {}, {}, {}\n"
                                   .format(
                                           idx, b_idx+1, valid_loss,
                                           valid_BLEU,
                                           ', '.join(map(str, valid_ROUGE))))
                    print_time_info(
                            "Epoch {} batch {} (batch {}): \
                                    [VALID] loss = {}, \
                            BLEU = {}, ROUGE_(1,2,L,BE) = {}"
                            .format(
                                idx, b_idx+1, self.batches,
                                valid_loss, valid_BLEU,
                                ', '.join(map(str, valid_ROUGE))))

            if verbose_epochs == 0 or idx % verbose_epochs == 0:
                print_time_info(
                        "Epoch {}: [TRAIN] loss = {}, \
                        BLEU = {}, ROUGE = {}"
                        .format(
                            idx, epoch_loss, epoch_BLEU,
                            ', '.join(map(str, epoch_ROUGE))))

            # validation
            if idx % valid_epochs == 0:
                batch = next(iter(self.test_data_loader))
                valid_loss, valid_BLEU, valid_ROUGE = self.run_batch(
                        batch,
                        criterion,
                        curriculum_layers,
                        testing=True,
                        result_path=os.path.join(
                            os.path.join(self.log_dir, "validation"),
                            "curri_layer{}_epoch{}_result.txt"
                            .format(curriculum_layers, idx)
                        )
                )

                with open(self.valid_epoch_log_path, 'a') as file:
                    file.write("{}, {}, {}, {}\n".format(
                        idx, valid_loss, valid_BLEU,
                        ', '.join(map(str, epoch_ROUGE))))
                print_time_info(
                        "Epoch {}: [VALID] loss = {}, \
                        BLEU = {}, ROUGE = {}"
                        .format(
                            idx, valid_loss, valid_BLEU,
                            ', '.join(map(str, valid_ROUGE))))
                # tensorboard logging
                if self.logger:
                    self.logger.write_valid(
                            valid_loss, valid_BLEU, valid_ROUGE, idx)
            # save model
            if idx % save_epochs == 0:
                print_time_info("Epoch {}: save model...".format(idx))
                self.save_model(self.model_dir)

            _teacher_forcing_ratio *= tf_decay_rate
            _inner_teacher_forcing_ratio *= inner_tf_decay_rate
            _inter_teacher_forcing_ratio *= inter_tf_decay_rate

    def run_batch(
            self,
            batch,
            criterion,
            curriculum_layers,
            testing=False,
            split_teacher_forcing=False,
            teacher_forcing_ratio=0.5,
            inner_teacher_forcing_ratio=0.5,
            inter_teacher_forcing_ratio=0.5,
            schedule_sampling=False,
            inner_schedule_sampling=False,
            inter_schedule_sampling=False,
            max_norm=None,
            result_path=None
            ):
        """
        When testing=False, run_batch is in training mode, and you should
        pass the argument teacher_forcing_ratio and max_norm
        When testing=True, run_batch is in testing mode, you should pass
        the argument result_path to store the result of testing batch
        """
        if not testing:
            # Initialize the optimizers (when training)
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

        # Initialize the loss
        loss = 0
        all_loss = 0

        # Generate the inputs for encoder
        if len(batch) == 3:
            raw_encoder_input, decoder_labels, de_lengths = batch
        else:
            raw_encoder_input, decoder_labels, inter_labels, de_lengths = batch
        batch_size = len(raw_encoder_input)
        encoder_input = Variable(
                torch.LongTensor(raw_encoder_input),
                volatile=testing)
        encoder_input = encoder_input.cuda() if use_cuda else encoder_input
        encoder_hidden = (self.encoder.initAttrHidden(
            encoder_input, batch_size)
            if self.en_use_attr_init_state
            else self.encoder.initHidden(batch_size))

        # We need to initialize the cell states for LSTM
        if self.cell == "LSTM":
            encoder_cell = self.encoder.initHidden(batch_size)

        # TODO: We should get all hidden states from encoder,
        # not only the last one
        if self.encoder.cell == "GRU":
            encoder_outputs, encoder_hidden = self.encoder(
                    encoder_input, encoder_hidden)
        elif self.encoder.cell == "LSTM":
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder(
                    encoder_input, encoder_hidden, encoder_cell)

        # TODO: Let attention matrix be returned to make other use of it
        # TODO: Add layer-wise attention?
        # TODO: How to decide to do teacher forcing or not?

        # For first layer of decoder, we use the last output as input;
        # for the other layers, we use the output from previous layer as input

        # Prepare the space for results from decoder (when testing)
        if testing:
            decoder_results = [np.zeros(
                (batch_size, decoder_labels[idx].shape[1]))
                for idx in range(curriculum_layers)]

        if not testing:
            if not split_teacher_forcing:
                if schedule_sampling:
                    use_teacher_forcing = [
                                [
                                    True if random.random() <
                                    teacher_forcing_ratio
                                    else False
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                else:
                    _use_teacher_forcing = (
                            True if random.random() < teacher_forcing_ratio
                            else False)
                    use_teacher_forcing = [
                                [
                                    _use_teacher_forcing
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                use_inner_teacher_forcing = use_teacher_forcing
                use_inter_teacher_forcing = [[0]] + use_teacher_forcing[:-1]
            else:
                if inner_schedule_sampling:
                    use_inner_teacher_forcing = [
                                [
                                    True if random.random() <
                                    inner_teacher_forcing_ratio
                                    else False
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                else:
                    _use_inner_teacher_forcing = (
                            True
                            if random.random() < inner_teacher_forcing_ratio
                            else False)
                    use_inner_teacher_forcing = [
                                [
                                    _use_inner_teacher_forcing
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                if inter_schedule_sampling:
                    use_inter_teacher_forcing = [
                                [
                                    True if random.random() <
                                    inter_teacher_forcing_ratio
                                    else False
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                else:
                    _use_inter_teacher_forcing = (
                            True
                            if random.random() < inter_teacher_forcing_ratio
                            else False)
                    use_inter_teacher_forcing = [
                                [
                                    _use_inter_teacher_forcing
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers - 1)
                            ]
                    use_inter_teacher_forcing = \
                        [[0]] + use_inter_teacher_forcing
        else:
            use_teacher_forcing = [
                        [
                            False
                            for _ in range(decoder_labels[idx].shape[1])
                            ]
                        for idx in range(curriculum_layers)
                    ]
            use_inner_teacher_forcing = use_teacher_forcing
            use_inter_teacher_forcing = [[0]] + use_teacher_forcing[:-1]

        # BLEU/ROUGE scores
        bleu_scores = []
        rouge_scores = []

        """
            First layer: seq2seq
            Other layers: RNN (input from first layer output / labels)
        """
        # all_decoder_inputs: input from the last layer
        # real_decoder_inputs: actual input from the last layer
        # note that there is 'repeat input' mechanism
        all_decoder_inputs = [[] for _ in range(curriculum_layers)]
        real_decoder_inputs = [[] for _ in range(curriculum_layers)]
        decoder_inputs = None
        for d_idx, decoder in enumerate(self.decoders[:curriculum_layers]):
            # for recording actual inputs
            _real_decoder_inputs = Variable(
                    torch.LongTensor(batch_size, 1).fill_(_PAD))
            _real_decoder_inputs = (
                _real_decoder_inputs.cuda()
                if use_cuda else _real_decoder_inputs)
            # Prepare for initial hidden state and cell
            decoder_hidden = decoder.transform_layer(encoder_hidden)
            if self.cell == "LSTM":
                decoder_cell = encoder_cell

            # Prepare for first input of certain layer
            if d_idx == 0:
                # First input of first layer must be _BOS
                decoder_input = Variable(
                        torch.LongTensor(batch_size, 1).fill_(_BOS))
            else:
                if use_inter_teacher_forcing[d_idx][0]:
                    decoder_input = Variable(
                        torch.LongTensor(decoder_labels[d_idx-1][:, 0])) \
                            .unsqueeze(1)
                else:
                    decoder_input = decoder_inputs[:, 0].unsqueeze(1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            _real_decoder_inputs = torch.cat((
                _real_decoder_inputs, decoder_input), 1)
            # Set last_output of the first step to BOS
            last_output = Variable(
                   torch.LongTensor(batch_size, 1).fill_(_BOS)
            )

            # Prepare for input of next layer
            if d_idx < curriculum_layers - 1:
                next_decoder_inputs = Variable(
                        torch.LongTensor(
                            batch_size,
                            decoder_labels[d_idx+1].shape[1]).fill_(_PAD))
                next_decoder_inputs = (
                        next_decoder_inputs.cuda() if use_cuda
                        else next_decoder_inputs)

            # for calculating BLEU
            hypothesis = torch.LongTensor(batch_size, 1, 1).fill_(_PAD)
            hypothesis = hypothesis.cuda() if use_cuda else hypothesis

            # Decoding sequence
            # for repeat_input, remember the offset of the label.
            label_idx = [0 for _ in range(batch_size)]
            for idx in range(decoder_labels[d_idx].shape[1]):
                last_output = last_output.cuda() if use_cuda else last_output

                if decoder.cell == "GRU":
                    decoder_output, decoder_hidden = decoder(
                            decoder_input, decoder_hidden, encoder_outputs,
                            last_output=last_output)
                elif decoder.cell == "LSTM":
                    decoder_output, decoder_hidden, decoder_cell = decoder(
                            decoder_input, decoder_hidden,
                            encoder_outputs, decoder_cell,
                            last_output=last_output)

                target = Variable(
                        torch.from_numpy(
                            decoder_labels[d_idx][:, idx])).cuda()

                loss += criterion(decoder_output.squeeze(1), target)

                topv, topi = decoder_output.data.topk(1)
                hypothesis = torch.cat((hypothesis, topi), 1)
                if d_idx < curriculum_layers - 1:
                    if use_inter_teacher_forcing[d_idx+1][idx]:
                        next_decoder_inputs[:, idx] = \
                            Variable(torch.from_numpy(
                                decoder_labels[d_idx][:, idx])).cuda()
                    else:
                        next_decoder_inputs[:, idx] = topi

                if testing:
                    decoder_results[d_idx][:, idx] = \
                        topi.cpu().numpy().squeeze((1, 2))
                # Decide next input of decoder
                if idx != decoder_labels[d_idx].shape[1] - 1:
                    # input from last step
                    if use_inner_teacher_forcing[d_idx][idx+1]:
                        last_output = target.unsqueeze(1)
                    else:
                        last_output = Variable(topi).squeeze(1)
                    # input from last layer
                    if d_idx == 0:
                        decoder_input = Variable(topi).squeeze(1)
                    else:
                        if self.repeat_input:
                            decoder_input = np.zeros(
                                    (batch_size, 1), dtype=np.int64)
                            predicts = topi.cpu().numpy().squeeze((1, 2))
                            labels = decoder_inputs.data.cpu().numpy()
                            for b_idx in range(len(label_idx)):
                                #  print(labels[b_idx][label_idx[b_idx]])
                                if predicts[b_idx] == labels[b_idx][
                                        label_idx[b_idx]]:
                                    label_idx[b_idx] += 1
                                decoder_input[b_idx][0] = labels[b_idx][
                                        label_idx[b_idx]]
                            decoder_input = Variable(
                                        torch.LongTensor(decoder_input))
                        else:
                            if idx >= decoder_labels[d_idx-1].shape[1]:
                                decoder_input = Variable(
                                    torch.LongTensor(
                                        batch_size, 1).fill_(_PAD))
                            else:
                                decoder_input = \
                                    decoder_inputs[:, idx+1].unsqueeze(1)

                    decoder_input = (
                            decoder_input.cuda() if use_cuda
                            else decoder_input)

                    _real_decoder_inputs = torch.cat((
                        _real_decoder_inputs, decoder_input), 1)

            if d_idx < curriculum_layers - 1:
                decoder_inputs = next_decoder_inputs

                # record the layer inputs
                all_decoder_inputs[d_idx+1] = decoder_inputs.data.cpu().numpy()
                real_decoder_inputs[d_idx+1] = _real_decoder_inputs.data \
                    .cpu().numpy()[:, 1:]

            hypothesis = hypothesis.squeeze(2).cpu().numpy()[:, 1:]
            bleu_score = BLEU(decoder_labels[d_idx], hypothesis)
            avg_bleu = np.mean(bleu_score)
            bleu_scores.append(bleu_score)

            rouge_score = ROUGE(decoder_labels[d_idx], hypothesis)
            avg_rouge_1_2_l_be = np.mean(rouge_score, axis=0)
            rouge_scores.append(rouge_score)

            # to prevent the graph keeping growing bigger and resulting in OOM
            # compute the gradients every layer (when training)
            if not testing:
                loss.backward(retain_graph=True)
            all_loss += loss.data[0] / de_lengths[d_idx]
            loss = 0

        if not testing:
            clip_grad_norm(self.encoder_parameters, max_norm)
            self.encoder_optimizer.step()
            clip_grad_norm(self.decoder_parameters, max_norm)
            self.decoder_optimizer.step()

        else:
            # untokenize the sentence,
            # e.g. [100, 200, 300] -> ['trouble', 'fall', 'piece']
            encoder_input = [
                    self.data_engine.tokenizer.untokenize(sent, is_token=True)
                    for sent in raw_encoder_input]
            decoder_results = [
                    [self.data_engine.tokenizer.untokenize(sent)
                        for sent in decoder_result]
                    for decoder_result in decoder_results]
            decoder_labels = [
                    [self.data_engine.tokenizer.untokenize(sent)
                        for sent in decoder_label]
                    for decoder_label in decoder_labels]
            # decoder inputs
            real_decoder_inputs = [
                [self.data_engine.tokenizer.untokenize(sent)
                 for sent in real_decoder_input]
                for real_decoder_input in real_decoder_inputs
            ]

            all_decoder_inputs = [
                [self.data_engine.tokenizer.untokenize(sent)
                 for sent in all_decoder_input]
                for all_decoder_input in all_decoder_inputs
            ]

            # write test results into files
            with open(result_path, 'a') as file:
                for idx in range(batch_size):
                    file.write("---------\n")
                    file.write("Data {}\n".format(idx))
                    file.write("encoder input: {}\n\n".format(
                        ' '.join(encoder_input[idx])))
                    for d_idx in range(curriculum_layers):
                        file.write("decoder layer {}\n".format(d_idx))
                        if d_idx > 0:
                            file.write(
                                "input from the last layer:\n{}\n".format(
                                    ' '.join(all_decoder_inputs[d_idx][idx])))
                            file.write("actual input:\n{}\n".format(
                                ' '.join(real_decoder_inputs[d_idx][idx])))
                        file.write("prediction:\n{}\n".format(
                            ' '.join(decoder_results[d_idx][idx])))
                        file.write("labels:\n{}\n".format(
                            ' '.join(decoder_labels[d_idx][idx])))
                        file.write("BLEU score: {}\n".format(
                            str(bleu_scores[d_idx][idx])))
                        file.write("ROUGE_(1,2,L,BE): {}\n".format(
                            str(', '.join(
                                map(str, rouge_scores[d_idx][idx])))))
                        file.write("\n")
                    file.write("\n")

        return all_loss, avg_bleu, avg_rouge_1_2_l_be

    def save_model(self, model_dir):
        encoder_path = os.path.join(model_dir, "encoder.ckpt")
        decoder_paths = [
                os.path.join(model_dir, "decoder_{}.ckpt".format(idx))
                for idx in range(self.n_decoders)]
        torch.save(self.encoder, encoder_path)
        for idx, path in enumerate(decoder_paths):
            torch.save(self.decoders[idx], path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir):
        # Get the latest modified model (files or directory)
        files_in_dir = glob.glob(os.path.join(model_dir, "*"))
        latest_file = sorted(files_in_dir, key=os.path.getctime)[-2]
        print(latest_file)
        if os.path.isdir(latest_file):
            encoder_path = os.path.join(latest_file, "encoder.ckpt")
            decoder_paths = [
                    os.path.join(
                        latest_file, "decoder_{}.ckpt".format(idx))
                    for idx in range(self.n_decoders)]
        else:
            encoder_path = os.path.join(model_dir, "encoder.ckpt")
            decoder_paths = [(
                os.path.join(model_dir, "decoder_{}.ckpt".format(idx))
                for idx in range(self.n_decoders))]

        loader = True
        if not os.path.exists(encoder_path):
            loader = False
        else:
            encoder = torch.load(encoder_path)
        decoders = []
        for path in decoder_paths:
            if not os.path.exists(path):
                loader = False
            else:
                decoders.append(torch.load(path))

        if not loader:
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.encoder = encoder
            self.decoders = decoders
            if os.path.isdir(latest_file):
                print_time_info("Load model from {} successfully".format(
                    latest_file))
            else:
                print_time_info("Load model from {} successfully".format(
                    model_dir))
