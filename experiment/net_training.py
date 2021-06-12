import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

class EarlyStoppingByLoss:
    def __init__(self, patience=10, min_delta=0):
        self.previous_loss = None
        self.epochs_no_improvement = 0
        self.patience = patience
        self.min_delta = min_delta
    
    def report(self, loss):
        if self.is_improvement(loss):
            self.epochs_no_improvement = 0
            self.previous_loss = loss
        else:
            self.epochs_no_improvement += 1

    def should_stop(self):
        return self.epochs_no_improvement >= self.patience
    
    def is_improvement(self, loss):
        return self.previous_loss == None or loss < self.previous_loss - self.min_delta

class EarlyStoppingByAccuracy:
    def __init__(self, patience=10, min_delta=0):
        self.previous_accuracy = None
        self.epochs_no_improvement = 0
        self.patience = patience
        self.min_delta = min_delta
    
    def report(self, accuracy):
        if self.is_improvement(accuracy):
            self.epochs_no_improvement = 0
            self.previous_accuracy = accuracy
        else:
            self.epochs_no_improvement += 1

    def should_stop(self):
        return self.epochs_no_improvement >= self.patience
    
    def is_improvement(self, accuracy):
        return self.previous_accuracy == None or accuracy > self.previous_accuracy + self.min_delta
    

def train(net, train_dataloader, criterion, optimizer, device="cpu"):
    net.train()
    running_loss = 0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(train_dataloader.dataset)
    return {"loss":epoch_loss}

def train_inception(net, train_dataloader, criterion, optimizer, device="cpu"):
    net.train()
    running_loss = 0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs, aux_outputs = net(inputs)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4*loss2
        running_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(train_dataloader.dataset)
    return {"loss":epoch_loss}

def evaluate(net, test_dataloader, label_set, criterion, device="cpu"):
    net.eval()
    correct = 0
    total = 0
    correct_labels = {label_str: 0 for label_str in label_set.get_str_labels()}
    total_labels = {label_str: 0 for label_str in label_set.get_str_labels()}
    running_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_labels[label_set.int_to_str(label.item())] += 1
                total_labels[label_set.int_to_str(label.item())] += 1
    accuracy = correct/total
    label_accuracy = calculate_label_accuracy(correct_labels, total_labels)
    epoch_loss = running_loss / len(test_dataloader.dataset)
    return {"correct":correct, "total":total, "accuracy":accuracy, "loss":epoch_loss,
            "correct_labels":correct_labels, "total_labels":total_labels, "label_accuracy":label_accuracy}

def calculate_label_accuracy(correct_labels, total_labels):
    label_accuracy = {}
    for label_str, correct_label in correct_labels.items():
        total_label = total_labels[label_str]
        if total_label == 0:
            label_accuracy[label_str] = np.nan
        else:
            label_accuracy[label_str] = correct_label / total_label
    return label_accuracy

def print_epoch_results(epoch_results):
    print_epoch_results_unpacked(**epoch_results)

def print_epoch_results_unpacked(epoch,train_loss, val_loss, val_accuracy, elapsed_time):
    print(f"Epoch {epoch}:")
    print(f"\ttrain loss: {train_loss}")
    print(f"\tvalidation loss: {val_loss}, validation accuracy: {100 * val_accuracy}%")
    print(f"\tElapsed time: {elapsed_time}")

def print_final_results(final_results):
    final_loss = final_results["loss"]
    final_accuracy = final_results["correct"]/final_results["total"]
    print("Final loss: {:.4f}, final accuracy: {:.2f} %".format(final_loss, 100*final_accuracy))
    print_label_accuracy(final_results["correct_labels"], final_results["total_labels"])

def print_label_accuracy(correct_labels, total_labels):
    for label_str, correct_label in correct_labels.items():
        total_label = total_labels[label_str]
        if total_label != 0:
            accuracy = correct_label / total_label
            print("\t{}: {:.2f}%".format(label_str, 100*accuracy))

def train_and_evaluate(net, train_dataloader, test_dataloader, label_set, epochs=30,
                       optimizer=None, early_stopping=None, print_results=False, is_inception=False, device=None):
    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(net.parameters()) if optimizer == None else optimizer
    all_epoch_results = []
    if early_stopping == None:
        early_stopping = EarlyStoppingByAccuracy(patience = 20)
    for epoch in range(epochs):
        if early_stopping.should_stop():
            break
        start_time = datetime.datetime.now()
        if is_inception:
            train_results = train_inception(net, train_dataloader, criterion, optimizer, device=device)
        else:
            train_results = train(net, train_dataloader, criterion, optimizer, device=device)
        results = evaluate(net, test_dataloader, label_set, criterion, device=device)
        time_elapsed = datetime.datetime.now() - start_time
        train_loss = train_results["loss"]
        val_loss = results["loss"]
        accuracy = results["accuracy"]
        early_stopping.report(accuracy)
        epoch_results = {"epoch":epoch, "train_loss":train_loss, "val_loss":val_loss, "val_accuracy":accuracy,
                          "elapsed_time":time_elapsed}
        all_epoch_results += epoch_results
        if print_results:
            print_epoch_results(epoch_results)

    final_results = evaluate(net, test_dataloader, label_set, criterion, device=device)
    if print_results:
        print_final_results(final_results)
    return final_results
