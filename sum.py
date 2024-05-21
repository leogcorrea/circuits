import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logicc
from itertools import pairwise
import time

# Digit classification network definition.
class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d(1, 6, 5),
      torch.nn.MaxPool2d(2, 2),
      torch.nn.ReLU(True),
      torch.nn.Conv2d(6, 16, 5),
      torch.nn.MaxPool2d(2, 2),
      torch.nn.ReLU(True)
    )
    self.classifier = torch.nn.Sequential(
      torch.nn.Linear(16 * 4 * 4, 120),
      torch.nn.ReLU(),
      torch.nn.Linear(120, 84),
      torch.nn.ReLU(),
      torch.nn.Linear(84, 10),
      torch.nn.Softmax(1)
    )

  def forward(self, x):
    x = self.encoder(x)
    x = x.view(-1, 16 * 4 * 4)
    x = self.classifier(x)
    return x
  

  # Retrieve the MNIST data.
def mnist_data():
  train = datasets.MNIST(root = "../data", train = True, download = True)
  test  = datasets.MNIST(root = "../data", train = False, download = True)
  return train.data.float().reshape(len(train), 1, 28, 28)/255., train.targets, \
         test.data.float().reshape(len(test), 1, 28, 28)/255., test.targets

# Normalization function to center pixel values around mu with standard deviation sigma.
def normalize(X_R, Y_R, X_T, Y_T, mu, sigma):
  return (X_R-mu)/sigma, Y_R, (X_T-mu)/sigma, Y_T

# Whether to pick the first or second half of the dataset.
def pick_slice(data, digit):
  h = len(data)//2
  return slice(h, len(data)) if digit else slice(0, h)

# MNIST images for the train set.
def mnist_data_select(digit, dataset): return dataset[pick_slice(dataset, digit)]

# Observed atoms for training.
def mnist_labels(dataset):
  # We join the two halves (top and bottom) of MNIST and join them together to get
  # two digits side by side. The labels are atoms encoding the sum of the two digits.
  labels = torch.concatenate((dataset[:(h := len(dataset)//2)].reshape(-1, 1),
                              dataset[h:].reshape(-1, 1)), axis=1)
  return [[f"sum({x.item() + y.item()})"] for x, y in labels]


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    #for batch_idx, (data, target) in enumerate(zip(train_x, train_y)):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) #torch.tensor([target]).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #log_interval = 10
        #dry_run = False
    #if batch_idx % log_interval == 0:
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, (batch_idx-1) * len(data), len(train_loader.dataset),
        100. * (batch_idx-1) / len(train_loader), loss.item()))
    #if dry_run:
        #pass # break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    b = 1
    with torch.no_grad():
        for data, target in test_loader:
            print("\rBatch {} of {}".format(b, 10))
            b+=1
            #data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def operate(nn_model, pc_model, query_builder, device, data_loader):
    nn_model.eval()
    test_loss = 0
    correct = 0
    
    queries = []
    for k in range(0, 19):
        queries.append(query_builder( f"add({k})" ))
    s = len(data_loader.dataset)//2
    n = 19
    m = 20
    b = 1
    d = 0
    probs = torch.ones(500, circuit.nliterals).to(device)
    with torch.no_grad():
        
        for data, target in data_loader:
            start = time.time()
            print("\rBatch {} of {}".format(b, 10))
            b+=1
            data, target = data.to(device), target.to(device)
            output = nn_model(data)
            h = output.size(dim=0)//2

            pred = torch.empty(h).to(device)
            tgt = torch.add(target[:h], target[h:]).float().to(device)
            probs[:, 0:m] = torch.concat((output[:h], output[h:]), dim = 1)

            for i, (p, t) in enumerate(zip(probs, tgt)): 
                #probs[0:m] = torch.concat((prob1, prob2)) #.to(device)
                pc_model.set_input_weights(p)
                #expr = f"add({prob1.argmax(keepdim=True).item() + prob2.argmax(keepdim=True).item()})"
                #pred = pc_model.query(query_builder(expr))

                q = torch.empty(n)
                for k in range(0, n):
                   q[k] = pc_model.query(queries[k])
                pred[i] = q.argmax(keepdim=True)
            elapsed = time.time() - start
            print("Elapsed time:", elapsed)

            #test_loss += F.nll_loss(pred, tgt, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(pred, tgt)
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += tgt.eq(pred.view_as(tgt)).sum().item()
 
    test_loss /= s

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, s,
        100. * correct / (s)))

 

if __name__ == '__main__':
    epochs = 10
    use_cuda = False# torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    torch.manual_seed(1)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    

    #train_X, train_Y, test_X, test_Y = normalize(*mnist_data(), 0.1307, 0.3081)


    model.load_state_dict(torch.load("sum_cnn.pt"), strict=False)



    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    #for epoch in range(1,  epochs + 1):
    #     train(model, device, train_loader, optimizer, epoch)
    #     test(model, device, test_loader)
    #     scheduler.step()

    # save_model = True
    # if save_model:
    #     torch.save(model.state_dict(), "sum_cnn.pt")

    source = "/home/leogcorrea/code/circuits/digits.pasp"
    c2d_executable = "/home/leogcorrea/code/circuits/c2d_linux"

    filename, symbols = logicc.pasp2cnf(source)
    filename = logicc.cnf2nnf(filename, c2d_executable)
    circuit = logicc.build_circuit_from_file(filename)
    circuit.to(device)
    query_builder = lambda expr: logicc.make_query(expr, symbols)
    
    operate(model, circuit, query_builder, device, test_loader)
