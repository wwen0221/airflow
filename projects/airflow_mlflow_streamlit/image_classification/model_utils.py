from torchvision.models import resnet50, ResNet50_Weights
import torch
import mlflow
from datetime import datetime

def initiate_model(tl = True):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    if tl:
        for param in model.parameters():
            param.requires_grad = False

        model.fc = torch.nn.Sequential(
                torch.nn.Linear(2048, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, 971))

    model.to('mps')

    return model

def train_model(model, image_datasets, dataloaders, criterion, optimizer, num_epochs=3):
    with mlflow.start_run():
        print('training....')
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to('mps')
                    labels = labels.to('mps')

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.detach() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.float() / len(image_datasets[phase])
                mlflow.log_metric(f"{phase}_loss", epoch_loss)
                mlflow.log_metric(f"{phase}_accuracy", epoch_acc)


                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss.item(),
                                                            epoch_acc.item()))
        return model

def save_model_weights(model_trained):
    now = datetime.now()
    timestamp = now.strftime(f"%Y_%m_%d_%H_%M")
    torch.save(model_trained.state_dict(), f'/Users/WW/airflow/projects/airflow_mlflow_streamlit/models/image_classification/playing_card/{timestamp}.h5')

def load_tl_model(weights_path):
    model = resnet50()

    model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 971))
    
    model.load_state_dict(torch.load(weights_path))
    model.to('mps')
    return model

