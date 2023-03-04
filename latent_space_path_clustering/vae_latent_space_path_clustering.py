import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.cluster import KMeans, MiniBatchKMeans


class pre_processing:

    def __init__(self, datadir, n_samples, split_ratio=0.25, batch_size=500):
        self.datadir = datadir
        self.n_smaples = n_samples
        self.ratio = split_ratio
        self.batch_size = batch_size
        self._cutoff = 1e-10

    def __transform(self, img):
        return img > self._cutoff

    class __inner_distdataset(Dataset):
        def __init__(self, files, transform=None):
            self.files = files
            self._transform_ = transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img_path = self.files[idx]
            image = np.load(img_path, allow_pickle=True)
            if self._transform_:
                image = self._transform_(image)
            image = np.expand_dims(image, axis=0)
            return img_path, image

    def create_dataloader(self):
        self.__create_dataloader()
        return self.train_loader, self.val_loader

    def __create_dataloader(self):
        distfiles = list(glob.glob(os.path.join(self.datadir, '*.npy')))
        distfiles = sorted(distfiles, key=lambda x: int(os.path.basename(x).split('_')[1]))
        self.distfiles = distfiles[:self.n_smaples]
        np.random.shuffle(self.distfiles)
        trainfiles, valfiles = self.distfiles[:int(self.n_smaples * (1 - self.ratio))], self.distfiles[
                                                                                        int(self.n_smaples * (
                                                                                                    1 - self.ratio)):]
        train_dataset = self.__inner_distdataset(trainfiles, self.__transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = self.__inner_distdataset(valfiles, self.__transform)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self, len_vector, d_latent):
        super(VAE, self).__init__()
        ## encoder layers
        self.encfc1 = nn.Linear(len_vector, 500)
        self.encfc2 = nn.Linear(500, 100)
        self.encfc31 = nn.Linear(100, d_latent)
        self.encfc32 = nn.Linear(100, d_latent)
        ## decoder layers
        self.decfc3 = nn.Linear(d_latent, 100)
        self.decfc2 = nn.Linear(100, 500)
        self.decfc1 = nn.Linear(500, len_vector)

    def encode(self, x):
        h = x.view(-1, len_vector)
        h = F.relu(self.encfc1(h))
        h = F.relu(self.encfc2(h))
        return self.encfc31(h), self.encfc32(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decfc3(z))
        h = F.relu(self.decfc2(h))
        return torch.sigmoid(self.decfc1(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, c_weight):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -c_weight * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        raise ValueError("The created / required file has already been created;")


def train(epoch, model, len_vector, train_loader, optimizer, device, c_weight, ckpt_interval, paramdir):
    model.train()
    train_loss = 0
    train_loss_save = []
    for batch_idx, (files, data) in enumerate(train_loader):
        data = data.float().to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model.forward(data)
        loss = loss_function(recon_batch, data.view(-1, len_vector), mu, logvar, c_weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
    if epoch % ckpt_interval == 0:
        torch.save(model.state_dict(), \
                   os.path.join(paramdir, 'epoch_{}.ckpt'.format(epoch)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    #     writer.add_scalar('Loss/train', train_loss/len(train_loader.dataset), epoch)
    train_loss_save.append(train_loss / len(train_loader.dataset))
    return train_loss_save


def validate(epoch, model, len_vector, val_loader, device, c_weight):
    model.eval()
    test_loss = 0
    test_loss_save = []
    with torch.no_grad():
        for i, (files, data) in enumerate(val_loader):
            data = data.float().to(device)
            recon_batch, mu, logvar = model.forward(data)
            test_loss += loss_function(recon_batch, data.view(-1, len_vector), mu, logvar, c_weight).item()
    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    #     writer.add_scalar('Loss/validation', test_loss, epoch)
    test_loss_save.append(test_loss)
    return test_loss_save


class DistDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = np.load(img_path, allow_pickle=True)
        if self.transform:
            image = self.transform(image)
        image = np.expand_dims(image, axis=0)
        return img_path, image


def transform(img):
    return img > 1e-10


use_cuda = True
if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


###############################################################
datadir = ''
paths = np.load(
    "",
    allow_pickle=True)
flux = np.loadtxt(
    "")
# num of samples used for training
n_samples = 6000
split_ratio = 0.25
batch_size = 300
learning_rate = 8e-5
weight_decay = 0
len_vector = 7500
d_latent = 2
epochs = 100
c_weight = 1

resultdir = 'results'\
/train' + str(b + 1) + '_nsamples' + str(n_samples) + '_batchsize' + str(batch_size) + '_lr' + str(
    learning_rate) + '_c' + str(c_weight) + '/'
##############################################################

mkdir(resultdir)
paramdir = resultdir + 'parameter_save'
mkdir(paramdir)

data_process = pre_processing(datadir=datadir, n_samples=n_samples, split_ratio=split_ratio, batch_size=batch_size)
train_loader, val_loader = data_process.create_dataloader()

model = VAE(len_vector=len_vector, d_latent=d_latent).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

trainscore = {}
testscore = {}
for epoch in range(epochs):
    trainscore[epoch] = train(epoch, model, len_vector, train_loader, optimizer, device, c_weight, ckpt_interval=20,
                              paramdir=paramdir)
    testscore[epoch] = validate(epoch, model, len_vector, val_loader, device, c_weight)
torch.save(model.state_dict(), os.path.join(paramdir, 'final_params.ckpt'))
np.save(resultdir + "/training_scores_save.npy", trainscore)
np.save(resultdir + "/testing_scores_save.npy", testscore)

distfiles = list(glob.glob(os.path.join(datadir, '*.npy')))
distfiles = sorted(distfiles, key=lambda x: int(os.path.basename(x).split('_')[1]))
distfiles = distfiles[:n_samples]
model.load_state_dict(torch.load(os.path.join(paramdir, 'final_params.ckpt')))
model.eval()
hidden_vectors = None
whole_dataset = DistDataset(distfiles, transform)
whole_loader = DataLoader(whole_dataset,
                          batch_size=len(whole_dataset), shuffle=False)
with torch.no_grad():
    for i, (files, data) in enumerate(whole_loader):
        data = data.float().to(device)
        mus = model.encode(data)
        if hidden_vectors is None:
            hidden_vectors = mus
        else:
            hidden_vectors = np.hstack([hidden_vectors, mus])
np.save(resultdir + "/trained_hidden_vectors_nsamples{}_batchsize{}_lr{}_c{}.npy".format(n_samples, batch_size,
                                                                                         learning_rate, c_weight),
        hidden_vectors)

##### Do clustering at latent space

n_clusters = 3

centers = hidden_vectors[0][:n_samples]
km_cluster = KMeans(n_clusters=n_clusters, max_iter=2000, random_state=42)
km_cluster.fit(centers)
np.savetxt(resultdir + "/{}paths_" + str(n_clusters) + "kmeans_clustering_labels.txt".format(n_samples),
           km_cluster.labels_)

km_lump = np.zeros(n_clusters)
for i in range(n_samples):
    km_lump[int(km_cluster.labels_[i])] += flux[i]

km_lump = km_lump / np.sum(km_lump)
np.savetxt(resultdir + "/"+str(n_samples)+"paths_" + str(n_clusters) + "kmeans_clustering_channel_flux.txt".format(n_samples),
           km_lump)

color = ['red', 'orange', 'springgreen', 'blue']
fig1 = plt.figure(dpi=600, figsize=(8, 8))
ax = fig1.add_subplot(111)
for i in range(n_samples):
    plt.scatter(centers[i, 0], centers[i, 1], c=color[int(km_cluster.labels_[i])])
ax.spines['bottom'].set_linewidth(2.0)
ax.spines['left'].set_linewidth(2.0)
ax.spines['top'].set_linewidth(2.0)
ax.spines['right'].set_linewidth(2.0)
plt.title('kmeans: ' + str(km_lump) + '\n nsamples' + str(n_samples) + '_batchsize' + str(batch_size) + '_lr' + str(
    learning_rate))
plt.savefig(resultdir + "/hidden_layer_" + str(n_clusters) + "kmeans_centers.png", bbox_inches='tight')