import torch
from torch import nn
from utilz.utils import *
from models.modules import *

def train_model(model, optimizer, criterion, scheduler, train_loader, test_loader, config):
    print("Training the model ...")
    patience_limit = config['patience_limit']
    sos = config['sos']
    eos = config['eos']
    train_loss, prev_average_loss, patience= [],  10000000.0, 0
    trainFDE, trainADE = [], []
    Best_Model = []
    for epoch in range(config['epochs']):
        model.train()
        epoch_losses, epoch_ADE, epoch_FDE = [], [], []
        for Scene, Target, Adj_Mat_Scene in train_loader: # Scene & Taget => [B, SL0, Nnodes, Features], Adj_Mat_Scene => [B, SL, Nnodes, Nnodes]
            optimizer.zero_grad()
            Scene, Scene_mask, Adj_Mat, Target = prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config)
            outputs = model(Scene, Scene_mask, Adj_Mat)
            loss = criterion(outputs.reshape(-1, 1024), Target.reshape(-1).long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            optimizer.step()
            with torch.no_grad():
                _, ADE, FDE = Find_topk_selected_words(outputs[:,1:-1], Target[:,1:-1]) # sos and eos are not included in the loss
            epoch_losses.append(loss.item())
            epoch_ADE.append(ADE)
            epoch_FDE.append(FDE)

        trainADE.append(sum(epoch_ADE)/ len(train_loader))
        trainFDE.append(sum(epoch_FDE)/ len(train_loader))
        train_loss.append(sum(epoch_losses) / len(train_loader))
        scheduler.step()
        
        

        log = f'Epoch [{epoch+1}/{config['epochs']}], Loss: {train_loss[-1]:.2f}, ADE: {trainADE[-1]:.3f}, FDE: {trainFDE[-1]:.2f}'
        savelog(log, config['ct'])
        # Saving the best model to the file
        if train_loss[-1] < prev_average_loss: # checkpoint update
            prev_average_loss = train_loss[-1]  # Update previous average loss
            patience = 0
            Best_Model = model
        elif patience > patience_limit:
                savelog(f'early stopping, Patience lvl1 , lvl2 {patience}', config['ct'])
                break
        patience += 1
        if config['Test_during_training'] and epoch % 5 == 3:
            _, ADE, FDE = test_model(model, test_loader, config)
            savelog(f"During Training, Test ADE: {ADE :.2f}, FDE: {FDE :.2f}", config['ct'])
            model.train()
            # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.5f} MB")
            # print(f"Reserved memory:  {torch.cuda.memory_reserved() / 1024**2:.5f} MB")
    return Best_Model, train_loss, trainADE, trainFDE


def test_model(model, test_loader, config):
    sos = config['sos']
    eos = config['eos']
    savelog("Starting testing phase", config['ct'])
    model.eval()
    Avg_ADE, Avg_FDE, test_size = 0, 0, 0
    with torch.no_grad():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for Scene, Target, Adj_Mat_Scene in test_loader:
            Scene, Scene_mask, Adj_Mat, Target = prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config)
            Pred_target = model(Scene, Scene_mask,Adj_Mat)
            Target = Target[:,1:-1,:, config['xy_indx']]
            Topk_Selected_words, ADE, FDE = Find_topk_selected_words(Pred_target[:,1:-1], Target)
            test_size += Scene.size(0)
            Avg_ADE += ADE
            Avg_FDE += FDE
        end_event.record()
        torch.cuda.synchronize()
        Avg_inference_time = start_event.elapsed_time(end_event)/test_size
        Avg_ADE, Avg_FDE = Avg_ADE/len(test_loader), Avg_FDE/len(test_loader)
        log= f"ADE is : {Avg_ADE:.3f} px \n FDE is: {Avg_FDE:.3f} px \n Inference time: {1000*Avg_inference_time:.3f} ms"
        savelog(log, config['ct'])
        return Topk_Selected_words, Avg_ADE, Avg_FDE
    
def prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config):
    Scene = attach_sos_eos(Scene, sos, eos)
    Adj_Mat = torch.cat((torch.ones_like(Adj_Mat_Scene[:,:1]), Adj_Mat_Scene, torch.ones_like(Adj_Mat_Scene[:,:1])), dim=1)
    Scene_mask = create_src_mask(Scene)
    Target = attach_sos_eos(Target[:,:,:, config['xy_indx']], sos[:, config['xy_indx']], eos[:,config['xy_indx']])
    return Scene, Scene_mask, Adj_Mat, Target
if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")