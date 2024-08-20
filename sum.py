import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logicc
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
  

# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device) 
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         #log_interval = 10
#     #if batch_idx % log_interval == 0:
#     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#         epoch, (batch_idx-1) * len(data), len(train_loader.dataset),
#         100. * (batch_idx-1) / len(train_loader), loss.item()))



# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     b = 1
#     with torch.no_grad():
#         for data, target in test_loader:
#             print("\rBatch {} of {}".format(b, 10))
#             b+=1
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
    

def operate(nn_model, pc_model, query_builder, device, data_loader, optimizer, epoch, train_mode = True):
    if not train_mode:
        nn_model.train()
    else:
        nn_model.eval()
    test_loss = 0
    #loss = 0
    correct = 0
    n = 19    
    queries = torch.zeros(n, 1)
    for k in range(0, n):
        queries[k] = query_builder( f"add({k})" )
    half_batch = data_loader.batch_size // 2
    s = len(data_loader.dataset)//2
    b = 1
    batches = len(data_loader.dataset) // data_loader.batch_size

    probs = torch.ones(half_batch, circuit.nliterals).to(device)

    ###f = torch.vmap(pc_model.query)

    with torch.enable_grad() if train_mode else torch.no_grad():        
        for batch_idx, (data, target) in enumerate(data_loader):
            if train_mode:
                optimizer.zero_grad()
            #start = time.time()
            print("\rBatch {} of {}".format(b, batches))
            b+=1
            data, target = data.to(device), target.to(device)
            output = nn_model(data)

            h = output.size(dim=0)//2

            #pred = torch.zeros(h, n).to(device)
            #tgt = torch.add(target[:h], target[h:]).to(device)
            pred = torch.empty(h).to(device)
            tgt = torch.add(target[:h], target[h:]).float().to(device)
            probs[:, 0:(n+1)] = torch.concat((output[:h], output[h:]), dim = 1)

            for i, p in enumerate(probs): 
                q = torch.zeros(n).to(device)

                pc_model.set_input_weights(p)

                for k in range(0, n):
                   q[k] = pc_model.query(queries[k])
                #   pred[i, k] = pc_model.query(queries[k])
                
                pred[i] = q.argmax(keepdim=True)

            #elapsed = time.time() - start
            #print("Batch elapsed time:", elapsed)
            tgt.requires_grad = True
            pred.requires_grad = True
            
            #correct += tgt.eq(pred.view_as(tgt)).sum().item()
            loss = F.cross_entropy(pred, tgt)
            #loss = F.nll_loss(pred, tgt) #, reduction='sum').item()  # sum up batch loss
         

            if train_mode:
                loss.backward()
                optimizer.step()
            else:
                test_loss += loss

            #correct += tgt.eq(pred.view_as(tgt)).sum().item()
 
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx-1) * len(data), len(data_loader.dataset),
            100. * (batch_idx-1) / len(data_loader), loss.item()))
    
    if not train_mode:
        test_loss /= s

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           loss, correct, s,
           100. * correct / (s)))

 

if __name__ == '__main__':
    epochs = 10
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    torch.manual_seed(1)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': 500}
    test_kwargs = {'batch_size': 50}

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
    


    ### model.load_state_dict(torch.load("sum_cnn.pt"), strict=False)



    source = "/home/leogcorrea/code/circuits/digits.pasp"
    c2d_executable = "/home/leogcorrea/code/circuits/c2d_linux"

    filename, symbols = logicc.pasp2cnf(source)
    filename = logicc.cnf2nnf(filename, c2d_executable)
    circuit = logicc.build_circuit_from_file(filename)
    circuit.to(device)
    query_builder = lambda expr: logicc.make_query(expr, symbols)


    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1,  epochs + 1):
        #train(model, device, train_loader, optimizer, epoch)
        #test(model, device, test_loader)
        operate(model, circuit, query_builder, device, test_loader, optimizer, epoch)
        #operate(model, circuit, query_builder, device, test_loader, optimizer, epoch, train_mode=False)

        scheduler.step()


    
    

    save_model = True
    if save_model:
        torch.save(model.state_dict(), "sum_cnn.pt")