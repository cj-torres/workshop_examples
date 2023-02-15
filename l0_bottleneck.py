import torch, math, copy

class L0_Regularizer(torch.nn.Module):

    def __init__(self, original_module: torch.nn.Module, lam: float, weight_decay: float = 0,
                 temperature: float = 2 / 3, droprate_init=0.2, limit_a=-.1, limit_b=1.1, epsilon=1e-6
                 ):
        super(L0_Regularizer, self).__init__()
        self.module = copy.deepcopy(original_module)

        self.pre_parameters = torch.nn.ParameterDict(
            {name.replace(".", "#") + "_p": param for name, param in self.module.named_parameters()}
        )

        self.param_names = [name.replace(".", "#") for name, param in self.module.named_parameters()]
        self.mask_parameters = torch.nn.ParameterDict()
        self.lam = lam
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.droprate_init = droprate_init
        self.limit_a = limit_a
        self.limit_b = limit_b
        self.epsilon = epsilon

        for name, param in self.module.named_parameters():
            mask = torch.nn.Parameter(torch.Tensor(param.size()))
            self.mask_parameters.update({name.replace(".", "#") + "_m": mask})

        # below code guts the module of its previous parameters,
        # allowing them to be replaced by non-leaf tensors

        self.reset_parameters()

        for name in self.param_names:
            L0_Regularizer.recursive_del(self.module, name)
            L0_Regularizer.recursive_set(self.module, name, self.sample_weights(name, 1))

    ''' 
    Below code copy with adaptations from codebase for: 

    Louizos, C., Welling, M., & Kingma, D. P. (2017). 
    Learning sparse neural networks through L_0 regularization. 
    arXiv preprint arXiv:1712.01312.
    '''

    def reset_parameters(self):
        for name, weight in self.pre_parameters.items():
            if "bias" in name:
                torch.nn.init.constant_(weight, 0.0)
            else:
                torch.nn.init.xavier_uniform_(weight)

        for name, weight in self.mask_parameters.items():
            torch.nn.init.normal_(weight, math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

    def constrain_parameters(self):
        for name, weight in self.mask_parameters.items():
            weight.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x, param):
        """Implements the CDF of the 'stretched' concrete distribution"""
        # references parameters
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(
            logits * self.temperature - self.mask_parameters[param + "_m"]).clamp(min=self.epsilon,
                                                                                  max=1 - self.epsilon)

    def quantile_concrete(self, x, param):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        # references parameters
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.mask_parameters[param + "_m"]) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def _reg_w(self, param):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_l2 = - (.5 * self.weight_decay * self.pre_parameters[param + "_p"].pow(2)) - self.lam
        logpw = torch.sum((1 - self.cdf_qz(0, param)) * logpw_l2)

        return -logpw

    def regularization(self):
        r_total = torch.Tensor([])
        for param in self.param_names:
            device = self.mask_parameters[param + "_m"].device
            r_total = torch.cat([r_total.to(device), self._reg_w(param).unsqueeze(dim=0)])
        return r_total.sum()

    def count_l0(self):
        total = []
        for param in self.param_names:
            total.append(torch.sum(1 - self.cdf_qz(0, param)).unsqueeze(dim=0))
        return torch.cat(total).sum()

    def count_l2(self):
        total = []
        for param in self.param_names:
            total.append(self._l2_helper(param).unsqueeze(dim=0))
        return torch.cat(total).sum()

    def _l2_helper(self, param):
        return (self.sample_weights(param, False) ** 2).sum()

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        # Variable deprecated and removed
        eps = torch.rand(size) * (1 - 2 * self.epsilon) + self.epsilon
        return eps

    def sample_z(self, param, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        size = self.mask_parameters[param + "_m"].size()
        if sample:
            device = self.mask_parameters[param + "_m"].device
            eps = self.get_eps(size).to(device)
            z = self.quantile_concrete(eps, param)
            return torch.nn.functional.hardtanh(z, min_val=0, max_val=1)
        else:
            pi = torch.sigmoid(self.mask_parameters[param + "_m"])
            return torch.nn.functional.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    def sample_weights(self, param, sample=True):
        mask = self.sample_z(param, sample)
        return mask * self.pre_parameters[param + "_p"]

    def forward(self, x):
        """rewrite parameters (tensors) of core module and feedforward"""
        for param in self.param_names:
            L0_Regularizer.recursive_set(self.module, param, self.sample_weights(param, sample=self.training))

        return self.module(x)

    @staticmethod
    def recursive_get(obj, att_name):
        if "#" in att_name:
            first, last = att_name.split("#", 1)
            L0_Regularizer.recursive_get(getattr(obj, first), last)
        else:
            return getattr(obj, att_name)

    @staticmethod
    def recursive_set(obj, att_name, val):
        if "#" in att_name:
            first, last = att_name.split("#", 1)
            L0_Regularizer.recursive_set(getattr(obj, first), last, val)
        else:
            setattr(obj, att_name, val)

    @staticmethod
    def recursive_del(obj, att_name):
        if "#" in att_name:
            first, last = att_name.split("#", 1)
            L0_Regularizer.recursive_del(getattr(obj, first), last)
        else:
            delattr(obj, att_name)


class SimpleLSTM(torch.nn.Module):
    '''LSTM with an embedding layer and additional linear layer on the output'''
    def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_sz
        self.embedding = torch.nn.Embedding(alphabet_sz, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_sz,
            batch_first=True
        )
        self.final_layer = torch.nn.Linear(hidden_sz, final_layer_sz)
        self.final_transform = torch.nn.Tanh()
        self.out = torch.nn.Linear(final_layer_sz, alphabet_sz)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            torch.nn.init.normal_(p.data)

    def forward(self, x):
        embeds = self.embedding(x)

        seq, out_h = self.lstm(embeds)
        final = self.final_transform(self.final_layer(seq))
        out = self.out(final)

        return out.squeeze()


def train(model, language_set, batch_sz, epochs, beta, sub_batch=8):
    from math import ceil
    from random import sample

    model = L0_Regularizer(model, beta)

    ce = torch.nn.CrossEntropyLoss()

    batches = len(language_set.train_input)
    batch_q = ceil(batches/batch_sz)
    batch_r = batches % batch_sz
    indxs = [i*batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_train = language_set.train_input
    y_train = language_set.train_output
    mask_train = language_set.train_mask

    op = torch.optim.Adam(model.parameters(), lr=.03)

    indices = range(batches)
    for epoch in range(epochs):
        for i in range(batch_sz // sub_batch):
            op.zero_grad()
            batch = torch.tensor(sample(indices, 8)).type(torch.LongTensor)
            model.train()
            x = x_train[batch]
            y = y_train[batch]
            mask = mask_train[batch]
            y_hat = model(x)

            ce_loss = ce(y_hat[mask], y[mask])
            re_loss = model.regularization()
            loss = ce_loss + re_loss
            loss.backward()
            del mask, x, y

            op.step()
            model.constrain_parameters()

        with torch.no_grad():
            model.eval()

            x_test = language_set.test_input
            y_test = language_set.test_output

            y_test_hat = model(x_test)
            test_mask = language_set.test_mask
            loss_test = ce(y_test_hat[test_mask], y_test[test_mask])
            size = model.count_l0()
        print("Epoch: %d, loss: %s, network size: %s" % (epoch+1, loss_test.item(), size.item()))


# 1 - element test
def test_1(n):

    in_tensors = torch.randn(500, n)
    target = torch.zeros_like(in_tensors)
    target[:, 0] = in_tensors[:, 0]

    model = L0_Regularizer(torch.nn.Linear(n, n), .01)
    opt = torch.optim.Adam(model.parameters())
    model.train()

    for i in range(10000):
        opt.zero_grad()
        loss = ((model(in_tensors) - target).pow(2)).mean()
        re_loss = model.regularization()

        total = loss + re_loss

        total.backward()
        opt.step()
        if i%100 == 0:
            print(loss.item())
            print(model.count_l0().item())

    model.eval()
    print((model(in_tensors) - target).pow(2).mean().item())


# all-element test (n^2)
def test_2(n):

    in_tensors = torch.randn(500, n)
    target = in_tensors.sum(dim=1).unsqueeze(dim=1).expand(-1, n)

    model = L0_Regularizer(torch.nn.Linear(n, n), .01)
    model.train()
    opt = torch.optim.Adam(model.parameters())

    for i in range(10000):
        opt.zero_grad()
        loss = ((model(in_tensors) - target).pow(2)).mean()
        re_loss = model.regularization()

        total = loss + re_loss

        total.backward()
        opt.step()
        if i%100 == 0:
            print(loss.item())
            print(model.count_l0().item())

    model.eval()
    print((model(in_tensors) - target).pow(2).mean().item())


# identity test (n)
def test_3(n):

    in_tensors = torch.randn(500, n)
    target = in_tensors

    model = L0_Regularizer(torch.nn.Linear(n, n), .01)
    opt = torch.optim.Adam(model.parameters())

    model.train()

    for i in range(10000):
        opt.zero_grad()
        loss = ((model(in_tensors) - target).pow(2)).mean()
        re_loss = model.regularization()

        total = loss + re_loss

        total.backward()
        opt.step()
        if i%100 == 0:
            print(loss.item())
            print(model.count_l0().item())

    model.eval()
    print((model(in_tensors) - target).pow(2).mean().item())


import language_builders as lb
anbm = lb.make_anbm_sets(1000, .05, .8, 175)
model = SimpleLSTM(4, 2, 4, 4)
train(model, anbm, 256, 1000, .01)
