#This file contains a bunch of DL network definitions that are used by the tuple embedding models
# 这里主要介绍了如何定义深度模型，以及如何训练它们，目前我们用不到
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from configurations import *
from torch.nn import functional as F


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

#This is a simple dataset for loading numpy matrices
class NumPy_Dataset(Dataset):
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

    def __getitem__(self, index):
        return torch.tensor(self.embedding_matrix[index, :]).float()

    def __len__(self):
        return len(self.embedding_matrix)

#This is used for CTT and Hybrid where you want to load L_e, R_e and y_e
# where L_e and R_e are the numpy matrices for tuple pairs 
# that are either a perturbation of each other as denoted by y_e
class NumPy_Triplet_Dataset(Dataset):
    def __init__(self, left_embedding_matrix, right_embedding_matrix, labels):
        self.left_embedding_matrix = left_embedding_matrix
        self.right_embedding_matrix = right_embedding_matrix
        self.labels = labels
        if (len(left_embedding_matrix) != len(right_embedding_matrix)) \
            or (len(right_embedding_matrix) != len(labels)):
            raise Exception("The dimensions of left and right embedding matrix and labels do not match")

    def __getitem__(self, index):
        left = torch.tensor(self.left_embedding_matrix[index, :]).float()
        right = torch.tensor(self.right_embedding_matrix[index, :]).float()
        label = torch.tensor(self.labels[index]).float()
        return left, right, label

    def __len__(self):
        return len(self.left_embedding_matrix)

#CTT

class CTTModel(nn.Module):
    # This model is assumed to be layered
    def __init__(self, input_dimension, hidden_dimensions):
        super(CTTModel, self).__init__()
        self.siamese_summarizer = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimensions[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dimensions[0], hidden_dimensions[1]),
            nn.ReLU(True)
        )
        # Simple Binary classifier
        self.classifier = nn.Linear(hidden_dimensions[1], 1)

    def forward(self, t1, t2):
        t1 = self.siamese_summarizer(t1)
        t2 = self.siamese_summarizer(t2)
        pred = self.classifier(torch.abs(t1 - t2))
        return torch.sigmoid(pred)

    def get_tuple_embedding(self, t1):
        with torch.no_grad():
            return self.siamese_summarizer(t1).detach().numpy()


class CTTModelTrainer:
    def __init__(self, input_dimension, hidden_dimensions):
        super(CTTModelTrainer, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions

    def train(self, left_embedding_matrix, right_embedding_matrix, labels,
              num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        self.model = CTTModel(self.input_dimension, self.hidden_dimensions)
        self.device = get_device()
        self.model.to(self.device)
        num_tuples = len(left_embedding_matrix)

        dataset = NumPy_Triplet_Dataset(left_embedding_matrix, right_embedding_matrix, labels)
        train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.model.train()

        for epoch in range(num_epochs):
            train_loss = 0
            for batch_idx, (left, right, label) in enumerate(train_dataloader):
                left = left.to(self.device)
                right = right.to(self.device)
                label = label.unsqueeze(-1)
                label = label.to(self.device)
                optimizer.zero_grad()
                output = self.model(left, right)
                loss = loss_function(output, label)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            # print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_tuples))

        self.model.eval()

        return self.model

    def save_model(self, output_file_name):
        torch.save(self.model.state_dict(), output_file_name)

    def load_model(self, input_file_name):
        self.model = CTTModel(self.input_dimension, self.hidden_dimensions)
        self.model.load_state_dict(torch.load(input_file_name))
        self.model.eval()


#deepBokcing的AE
class DB_AutoEncoder(nn.Module):
    # This model is assumed to be layered
    def __init__(self, input_dimension, hidden_dimensions):
        super(DB_AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimensions[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dimensions[0], hidden_dimensions[1])
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dimensions[1], hidden_dimensions[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dimensions[0], input_dimension)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_tuple_embedding(self, t1):
        with torch.no_grad():
            t1 = t1.to(get_device())
            a = self.encoder(t1).detach().cpu().numpy()
            # print(a)
            # print(type(a))
            return a


class DB_AutoEncoderTrainer:
    def __init__(self, input_dimension, hidden_dimensions):
        super(DB_AutoEncoderTrainer, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions

    def train(self, embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        self.model = DB_AutoEncoder(self.input_dimension, self.hidden_dimensions)
        self.device = get_device()
        self.model.to(self.device)
        num_tuples = len(embedding_matrix)

        train_dataloader = DataLoader(dataset=NumPy_Dataset(embedding_matrix), batch_size=batch_size, shuffle=True)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.model.train()

        for epoch in range(num_epochs):
            train_loss = 0
            for batch_idx, data in enumerate(train_dataloader):
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_function(output, data)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            # print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_tuples))

        self.model.eval()

        return self.model

    def save_model(self, output_file_name):
        torch.save(self.model.state_dict(), output_file_name)

    def load_model(self, input_file_name):
        self.model = AutoEncoder(self.input_dimension, self.hidden_dimensions)
        self.model.load_state_dict(torch.load(input_file_name))
        self.model.eval()


class AutoEncoder(nn.Module):
    #This model is assumed to be layered
    def __init__(self, input_dimension, hidden_dimensions):
        super(AutoEncoder, self).__init__()
        if hidden_dimensions<200:
            #200维以内的
            self.encoder = nn.Sequential(
                nn.Linear(input_dimension, 1000),
                nn.ReLU(True),
                nn.Linear(1000, 500),
                nn.ReLU(True),
                nn.Linear(500,250),
                nn.ReLU(True),
                nn.Linear(250,hidden_dimensions)
                )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dimensions, 250),
                nn.ReLU(True),
                nn.Linear(250, 500),
                nn.ReLU(True),
                nn.Linear(500,1000),
                nn.ReLU(True),
                nn.Linear(1000,input_dimension)
                )

        else:
            #500维以上的
            self.encoder = nn.Sequential(
                nn.Linear(input_dimension, 2*hidden_dimensions),
                nn.ReLU(True),
                nn.Linear(2*hidden_dimensions, hidden_dimensions)
                )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dimensions, 2*hidden_dimensions),
                nn.ReLU(True),
                nn.Linear(2*hidden_dimensions,input_dimension)
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
 
    def get_tuple_embedding(self, t1):
        with torch.no_grad():
            return self.encoder(t1).detach().cpu().numpy()

#只所有维度只使用一个NN
class AutoEncoder_V2(nn.Module):
    # This model is assumed to be layered
    def __init__(self, input_dimension, hidden_dimensions):
        super(AutoEncoder_V2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, 2 * hidden_dimensions),
            nn.ReLU(True),
            nn.Linear(2 * hidden_dimensions, hidden_dimensions)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dimensions, 2 * hidden_dimensions),
            nn.ReLU(True),
            nn.Linear(2 * hidden_dimensions, input_dimension)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_tuple_embedding(self, t1):
        with torch.no_grad():
            return self.encoder(t1).detach().cpu().numpy()


class AutoEncoderTrainer:
    def __init__(self, input_dimension, hidden_dimensions):
        super(AutoEncoderTrainer, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions

    def train(self, embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        self.model = AutoEncoder(self.input_dimension, self.hidden_dimensions)
        self.device = get_device()
        self.model.to(self.device)
        num_tuples = len(embedding_matrix)

        train_dataloader = DataLoader(dataset=NumPy_Dataset(embedding_matrix), batch_size=batch_size, shuffle=True)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)

        self.model.train()

        for epoch in range(num_epochs):
            train_loss = 0
            for batch_idx, data in enumerate(train_dataloader):
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_function(output, data)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if epoch ==0:
                print('====> Epoch: {} loss: {:.4f}'.format(epoch, train_loss))

        print("====> 最后的损失为：", train_loss)
        self.model.eval()
        
        return self.model
    
    def save_model(self, output_file_name):
        torch.save(self.model.state_dict(), output_file_name)

    def load_model(self, input_file_name):
        self.model = AutoEncoder(self.input_dimension, self.hidden_dimensions)
        self.model.load_state_dict(torch.load(input_file_name))
        self.model.eval()


class AutoEncoder_V2Trainer:
    def __init__(self, input_dimension, hidden_dimensions):
        super(AutoEncoder_V2Trainer, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions

    def train(self, embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        self.model = AutoEncoder_V2(self.input_dimension, self.hidden_dimensions)
        self.device = get_device()
        self.model.to(self.device)
        num_tuples = len(embedding_matrix)

        train_dataloader = DataLoader(dataset=NumPy_Dataset(embedding_matrix), batch_size=batch_size, shuffle=True)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.model.train()

        for epoch in range(num_epochs):
            train_loss = 0
            for batch_idx, data in enumerate(train_dataloader):
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_function(output, data)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if epoch == 0:
                print('====> Epoch: {} loss: {:.4f}'.format(epoch, train_loss))
            print('====> Epoch: {} loss: {:.4f}'.format(epoch, train_loss))
        print("====> 最后的损失为：", train_loss)
        self.model.eval()

        return self.model

    def save_model(self, output_file_name):
        torch.save(self.model.state_dict(), output_file_name)

    def load_model(self, input_file_name):
        self.model = AutoEncoder(self.input_dimension, self.hidden_dimensions)
        self.model.load_state_dict(torch.load(input_file_name))
        self.model.eval()


#VAE降维
class VAE(nn.Module):
    # This model is assumed to be layered
    def __init__(self, input_dimension, hidden_dimensions):
        super(VAE, self).__init__()
        self.hidden_dimensions = hidden_dimensions

            # 768->x-x->768->768
            # encoder
        self.fc1 = nn.Linear(input_dimension, hidden_dimensions)
        self.fc21 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.fc22 = nn.Linear(hidden_dimensions, hidden_dimensions)
            # decoder
        self.fc3 = nn.Linear(hidden_dimensions, input_dimension)
        self.fc4 = nn.Linear(input_dimension, input_dimension)

    # 编码  学习高斯分布均值与方差
    def encode(self,x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1),self.fc22(h1)

    # 将高斯分布均值与方差参数重表示，生成隐变量z  若x~N(mu, var*var)分布,则(x-mu)/var=z~N(0, 1)分布
    def reparaeterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return  mu+eps*std

    def docoder(self,z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)


    def forward(self, x):
        mu,logvar = self.encode(x)
        z = self.reparaeterize(mu=mu,logvar=logvar)
        return self.docoder(z),mu,logvar


    def get_tuple_embedding(self, t1):
        with torch.no_grad():
            mu,logvar = self.encode(t1)
            z = self.reparaeterize(mu=mu,logvar=logvar)

            return z.detach().cpu().numpy()


def VAE_loss_function(recon_x,x,mu,logvar):
    MSE = nn.MSELoss()
    MSE_loss = MSE(recon_x,x)
    KLD= -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return 0.1*KLD + MSE_loss


class VAETrainer:
    def __init__(self, input_dimension, hidden_dimensions):
        super(VAETrainer, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions

    def train(self, embedding_matrix, num_epochs=VAE_NUM_EPOCHS, batch_size=BATCH_SIZE):
        self.model = VAE(self.input_dimension, self.hidden_dimensions)
        self.device = get_device()
        self.model.to(self.device)
        num_tuples = len(embedding_matrix)

        train_dataloader = DataLoader(dataset=NumPy_Dataset(embedding_matrix), batch_size=batch_size, shuffle=True)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=VAE_LEARNING_RATE)

        self.model.train()

        for epoch in range(VAE_NUM_EPOCHS):
            train_loss = 0
            for batch_idx, data in enumerate(train_dataloader):
                data = data.to(self.device)
                optimizer.zero_grad()
                output,mu,logvar = self.model(data)
                loss = VAE_loss_function(recon_x=output,x=data,mu=mu,logvar=logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if epoch == 0:
                print('====> Epoch: {} loss: {:.4f}'.format(epoch, train_loss))
            print('====> Epoch: {} loss: {:.4f}'.format(epoch, train_loss))
            

        print("====> 最后的损失为：", train_loss)
        self.model.eval()

        return self.model

    def save_model(self, output_file_name):
        torch.save(self.model.state_dict(), output_file_name)

    def load_model(self, input_file_name):
        self.model = AutoEncoder(self.input_dimension, self.hidden_dimensions)
        self.model.load_state_dict(torch.load(input_file_name))
        self.model.eval()




