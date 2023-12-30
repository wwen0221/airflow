import torch

def eval_model():
    model.eval()

    with torch.no_grad():
        n_correct=0
        n_samples=0
        
        n_class_correct = [0 for i in range(len(classes))]
        n_class_sample = [0 for i in range(len(classes))]
        
        for inputs, labels in test_loader:
            # Send to device
            inputs = inputs.to('mps')
            labels = labels.to('mps')
        
            # Forward pass
            outputs = model(inputs)
            
            # Predictions
            _, preds = torch.max(outputs, 1)
            
            n_samples += labels.shape[0]
            n_correct += (preds == labels.data).sum().item()
            
            for i in range(CFG['batch_size']):
                try:
                    label = labels[i].item()
                    pred = preds[i].item()
                except:
                    break
                
                if (label==pred):
                    n_class_correct[label]+=1
                n_class_sample[label]+=1
        
        acc = 100 * n_correct/n_samples
        print(f'Overall accuracy on test set: {acc:.1f} %')
        
        for i in range(len(classes)):
            print(f'Accuracy of {classes[i]}: {100* n_class_correct[i]/n_class_sample[i]:.1f} %')