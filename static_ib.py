import torch

class GaussianSampler(torch.nn.Module):
    """
    Module to sample from multivariate Gaussian distributions.
    Converts input into a sampled vector of the same size.
    """

    def __init__(self, input_size: int):
        """
        :param input_size: size of vector that will be transformed to twice its length
        Variables:
        gauss_parameter_generator is a matrix that transforms input into a vector, first half of vector is the mean,
        second half the standard deviation
        """
        super(GaussianSampler, self).__init__()
        self.input_size = input_size
        self.gauss_parameter_generator = torch.nn.Linear(self.input_size, self.input_size*2)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def sample(self, stats):
        """
        Takes vector of parameters, transforms these into distributions using supported torch class
        Returns parameters and samples
        :param stats: vector of parameters
        :return: Tuple(Tensor of means, tensor of standard deviations, tensor of samples)
        """
        mu = stats[:, :, self.input_size:]
        std = torch.nn.functional.softplus(stats[:, :, :self.input_size], beta=1)
        std = torch.diag_embed(std)
        norm = torch.distributions.multivariate_normal.MultivariateNormal(mu, std)
        #eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        # output noisy sampling of size B x T x H
        return mu, std, norm.rsample()

    def forward(self, x: torch.Tensor):
        """
        Takes input vector x, produces samples
        :param x: Input tensor
        :return: Tuple(Tensor of samples, tensor of means, tensor of standard deviations)
        """
        # x is of size BxTxH
        gauss_parameters = self.gauss_parameter_generator(x)
        mu, std, sample = self.sample(gauss_parameters)
        return sample, mu, std


class GaussianLSTM(torch.nn.Module):
    """
    LSTM with a sampling layer on the output
    """
    def __init__(self, input_sz: int, hidden_sz: int, num_layers: int):
        super(GaussianLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size=self.input_sz,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.h_gauss_sampler = GaussianSampler(hidden_sz)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor,
                init_states = None):

        bs, seq_sz, _ = x.size()

        if init_states is None:
            h_t, c_t = (torch.stack([torch.zeros(self.num_layers, self.hidden_size).to(x.device)] * bs, dim=-2),
                        torch.stack([torch.zeros(self.num_layers, self.hidden_size).to(x.device)] * bs, dim=-2))
        else:
            h_t, c_t = init_states
        output, (_, _) = self.lstm(x, (h_t, c_t))
        h_seq, h_mus, h_stds = self.h_gauss_sampler(output)

        return h_seq, (h_mus, h_stds)


class SequentialVariationalIB(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_size):

        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.noise_size = hidden_size

        self.__build_model()

    def __build_model(self):
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )

        self.lstm = GaussianLSTM(self.embedding_dim, self.hidden_size, self.num_layers)

        self.initial_state = torch.nn.Parameter(torch.stack(
            [torch.randn(self.num_layers, self.hidden_size),
             torch.randn(self.num_layers, self.hidden_size)]
        ))

        self.decoder = torch.nn.Linear(self.noise_size, self.vocab_size)

    def get_initial_state(self, batch_size):
        init_a, init_b = self.initial_state
        return torch.stack([init_a] * batch_size, dim=-2), torch.stack([init_b] * batch_size, dim=-2)  # L x B x H

    def encode(self, X):
        # X is of shape B x T
        batch_size, seq_len = X.shape
        embedding = self.word_embedding(X)
        init_h, init_c = self.get_initial_state(batch_size)  # init_h is L x B x H
        h_seq, (h_mus, h_stds) = self.lstm(embedding, (init_h, init_c))  # output is B x T x H
        return h_seq, (h_mus, h_stds)

    def decode(self, h):
        logits = self.decoder(h)
        return logits

    def forward(self, x):
        h_seq, h_stats = self.encode(x)
        y_hat = self.decode(h_seq)

        return y_hat, h_stats

    def lm_loss(self, y, y_hat, mask):
        """
        Just the cross entropy loss
        :param y: Target tensor
        :param y_hat: Output tensor
        :param mask: Mask selecting only relevant entries for the loss
        :return: Cross entropy
        """
        ce = torch.nn.CrossEntropyLoss()
        # Y contains the target token indices. Shape B x T
        # Y_hat contains distributions
        ce_loss = ce(y_hat[mask], y[mask])
        return ce_loss

    def kl_loss(self, stats, mask):
        '''
        Calculates KL divergence between
        :param stats: Tuple(Mean, Std Deviations)
        :param mask: Mask of relevant entries
        :return: Mean KL divergence between distribution specified by stats and Z (a standard multivariate normal)
        '''
        std = stats[1]
        mu = stats[0]

        std_flat = std[mask]  # B*TxH remove padded entries
        mu_flat = mu[mask]  # B*TxH

        # z given x is a sequence of distributions
        zx_normals = torch.distributions.multivariate_normal.MultivariateNormal(mu_flat, std_flat)

        # z is a standard normal multivariate with the same dimensions as z given x
        z = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros_like(mu_flat),
                                                                       torch.diag_embed(torch.ones_like(mu_flat)))
        kld = torch.distributions.kl_divergence(zx_normals, z)

        mi = z.entropy()-zx_normals.entropy()

        return kld.mean(), mi.mean()


def train(epochs, beta):
    import language_builders as lb
    anbm = lb.make_anbn_sets(1000, .05, .8, 175)
    model = SequentialVariationalIB(4, 2, 1, 4)
    opt = torch.optim.Adam(model.parameters())
    for i in range(epochs):
        opt.zero_grad()
        y_hat, h_stats = model(anbm.train_input)
        ce_loss = model.lm_loss(anbm.train_output, y_hat, anbm.train_mask)
        kl_loss, mi = model.kl_loss(h_stats, anbm.train_mask)
        loss = ce_loss + beta * kl_loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            y_test_hat, _ = model(anbm.test_input)
            test_loss = model.lm_loss(anbm.test_output, y_test_hat, anbm.test_mask)
        print("CE Loss: {} KL Loss: {} Test Loss: {} MI: {}".format(ce_loss.item(), kl_loss.item(), test_loss.item(), mi.item()))
    return model


train(500, .05)