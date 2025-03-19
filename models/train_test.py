import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import os
from utilz.utils import *
import torch.nn.functional as F
import time
from models.HGAAGT import greedy

def train_model(model, optimizer, criterion, train_loader, test_loader, clip, args, datamax, datamin, ZoneConf):
    ct = args.ct
    epochs = args.epochs
    lr = args.learning_rate
    test_in_epoch = args.test_in_epoch
    patience_limit = args.patience_limit
    batch_size = args.batch_size
    columns_to_pick = args.Columns_to_Predict
    future = args.future
    input_size = args.input_size
    Nusers = args.Nusers
    sos = args.sos
    eos = args.eos
    prev_average_loss = 10000000.00000 #float('inf')  # Initialize with a high value
    epoch_losses = []
    test_loss = []
    scheduler = StepLR(optimizer, step_size=args.schd_stepzise, gamma=args.gamma)
    Max_Acc = 1000000 # collecting maximum accuracy so far
    patience = 0
    patienceLvl2 = 0
    dssc = args.dssc
    dstar = args.dstar
    for epoch in range(epochs):
        patience += 1
        model.train()
        train_loss = []
        loss_num = 0
        
        for Scene, Target, Adj_Mat_Scene in train_loader:
            loss_num += 1 #Scene.size(0)
            optimizer.zero_grad()
            scene = Scene[:,0::dssc,:,:input_size].clone().detach()
            scene = torch.cat((sos.repeat(scene.size(0),1,1,1), scene,eos.repeat(scene.size(0),1,1,1)), dim=1)
            adj_sos_eos = torch.ones_like(Adj_Mat_Scene[:,0]).unsqueeze(1) # This is done to match the scene size after eos and sos are added
            adj_mat = torch.cat((adj_sos_eos, Adj_Mat_Scene[:,0::dssc], adj_sos_eos), dim=1)
            scene_mask = create_src_mask(scene)
            target = Target[:, ::dstar,:, 1:3].clone().detach() #0:20:dstar

            target = torch.cat((sos[:,:,:,1:3].repeat(target.size(0),1,1,1), target,eos[:,:,:,1:3].repeat(target.size(0),1,1,1)), dim=1).long()
            target_in = target[:,:]
            target_out = target[:,:]
            trgt_mask = target_mask(target_in,args.n_heads)
            # trgt_mask = target_mask(target)
            # flg = target[:,-1:] !=0
            scene_mask = scene_mask.permute(0,2,1)
            src_mask = scene_mask.reshape(-1, scene.size(1))
            # trgt_mask = trgt_mask.permute(0,4,1,2,3)
            # trgt_mask = trgt_mask.reshape(-1, trgt_mask.size(-1), trgt_mask.size(-1))

            outputs = model(scene, src_mask,
                            adj_mat, target_in, trgt_mask) #torch.ones_like(target[:,-1:])*flg
            loss = criterion(outputs.reshape(-1, 1024), target_out.reshape(-1).long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss.append(float(loss.item()))
            # del scene, target, outputs  # Delete tensors
            # torch.cuda.empty_cache()
        # if epoch < 120 and epoch >220:
            # scheduler.step()
        scheduler.step()
        average_loss = sum(train_loss) / loss_num
        epoch_losses.append(average_loss)

        # Saving the best model to the file
        if average_loss < prev_average_loss: # checkpoint update
            checkpoint= {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint, ct, comment=f"Best Result at ct: {ct}, epoch: {epoch}")
            prev_average_loss = average_loss  # Update previous average loss
            patience = 0
            
            if test_in_epoch:
                predicteddata,Targetdata, l, acc, log = test_model(model,criterion, test_loader, columns_to_pick, future, Nusers, input_size, sos, eos,dssc, dstar)
                test_loss.append([l, epoch])
                savelog(log, ct)
                patienceLvl2 +=1
                if acc < Max_Acc:
                    patienceLvl2 = 0
                    Max_Acc = acc
                    Best_Model = model
                    Best_Predicteddata = predicteddata
                    log = f'Hit the Max Accuracy: {Max_Acc:.1f}, Epoch: {epoch+1}, Loss: {average_loss:.1f}, Learning rate: {lr:.5f}'
                    log = log + f' Batch: {batch_size}, ' # Hidden: {model.module.encoder.hidden_size}, Num_layers: {model.module.encoder.num_layers}'
                    savelog(log, ct)
        log = f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.5f}'
        savelog(log, ct)

        # Early stopping
        if patience > patience_limit or patienceLvl2 > patience_limit:
            savelog(f'early stopping, Patience lvl1 , lvl2 {patience}, and {patienceLvl2}', ct)
            break
        # if outputs.isnan().any():
        #     savelog('NaN values in the output', ct)
        #     Best_Model = []
        #     return [], [], [], [], 0
    if not test_in_epoch:
        Best_Predicteddata, Targetdata, l, acc, log = test_model(Best_Model,criterion, test_loader, columns_to_pick, future, Nusers, input_size, sos, eos, dssc, dstar)
        # predicteddata = torch.cat((predicteddata, p), dim=0)     
        test_loss.append([l, epoch])
        savelog(log, ct)
    return Best_Model, Best_Predicteddata, Targetdata, test_loss, epoch_losses, Max_Acc


def test_model(model,criterion, test_loader, columns_to_pick, future, Nusers, input_size, sos, eos,dssc, dstar):
    # savelog("Starting testing phase", ct)
    avg_inf_time = 0
    model.eval()
    test_losses = []
    test_loss_num = 0
    ll = 0   
    true_loss = 0
    lvl2_loss = 0
    with torch.no_grad():
 
        for Scene, Target, Adj_Mat_Scene in test_loader:
            scene = Scene[:,0::dssc,:,:input_size]
            scene = torch.cat((sos.repeat(scene.size(0),1,1,1), scene,eos.repeat(scene.size(0),1,1,1)), dim=1)
            scene_mask = create_src_mask(scene)
            scene_mask = scene_mask.permute(0,2,1)
            src_mask = scene_mask.reshape(-1, scene.size(1))
            adj_mat = Adj_Mat_Scene[:,0::dssc]
            target = Target[:,::dstar,:, 1:3].clone().long() #0:20:dstar
            # target = torch.cat((sos[:,:,:,1:3].repeat(target.size(0),1,1,1), target, eos[:,:,:,1:3].repeat(target.size(0),1,1,1)), dim=1)
            trgt_mask = target_mask(target[:,:])
            # trgt_mask = trgt_mask.permute(0,4,1,2,3)
            # trgt_mask = trgt_mask.reshape(-1, trgt_mask.size(-1), trgt_mask.size(-1))
            # flg = target[:,-1:] !=0
            # target = torch.cat((sos[:,:,:,1:3].repeat(target.size(0),1,1,1), target, eos[:,:,:,1:3].repeat(target.size(0),1,1,1)), dim=1)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            test_loss_num += 1 #Scene.size(0)
            ll += Scene.size(0)
            start_event.record()
            # Pred_target, avg_target = greedy(model, scene, adj_mat, sos[:,:,:,1:3].repeat(target.size(0),1,1,1), Nseq=20, device=target.device)
            Pred_target = model(scene, src_mask, adj_mat,target, trgt_mask)
            # loss = criterion(Pred_target.reshape(-1,10), target[:,-2,:,0].reshape(-1).long())
            # outputs = model(scene, scene_mask, adj_mat, target, trgt_mast, sos, future)

            end_event.record()
            torch.cuda.synchronize()

            true_loss += criterion(Pred_target[:,1:-1].reshape(-1, 1024), target.reshape(-1).long())
            # test_losses.append(tests_loss.item())
            a = Pred_target[:,1:-1].softmax(dim=-1)
            # true_loss += F.mse_loss(Pred_target,target)
            top_values, top_indices = torch.topk(a, k = 5, dim=-1)
            # true_loss += F.mse_loss(top_indices[:,:,:, 0], target[:, -1])
            b = (top_indices*top_values).sum(-1)/top_values.sum(-1)
            flg = target != 0
            lvl2_loss += torch.sqrt(torch.pow((b*flg -target),2).sum(-1)).mean()
            # lvl2_loss += F.mse_loss(avg_target, target)
            # if Trloss< LossThreshold:
            #     PredZones = zonefinder(outputs, ZoneConf)
            #     counts, totalzone, BtN, nonzero,doublezone = Zone_compare(PredZones, Target[:,-1,:32,6], Scene[:,-1,:32,6],outputs)
            #     CorrectZones += counts
            #     totalZones += totalzone
            #     totallen += BtN
            #     totalnonzero += nonzero
            #     totaldoublezone += doublezone
            try:
                Predicteddata = torch.cat((Predicteddata, Pred_target), dim=0)
                Targetdata = torch.cat((Targetdata, Target), dim=0)
            except:
                Predicteddata = Pred_target
                Targetdata = Target

            inference_time = start_event.elapsed_time(end_event)
            avg_inf_time += inference_time
        avg_test_loss = true_loss / test_loss_num
        # Calculate accuracy as the inverse of MSE
        accuracy = true_loss/test_loss_num  #100*A/C
        AccReal = lvl2_loss/test_loss_num
        
        avg_inf_time = avg_inf_time / ll
        log= f"Average Test Loss: {avg_test_loss:.3f} \n Accuracy %: {accuracy:.3f} \n Inference time: {1000*avg_inf_time:.3f} ms and Real Acc: {AccReal:.3f}"
        # if Trloss < LossThreshold:
        #     print(f"Total Zones: {totalZones}, Correct Zones: {CorrectZones}") 
        #     print(f"Acc: {CorrectZones/totalZones*100:.2f}, Single Zones: {totallen- totalZones}")
        #     print(f"totalnonzero: {totalnonzero}, non zer Acc: {CorrectZones/totalnonzero*100:.2f}")
        #     print(f"totaldoublezone: {totaldoublezone}")

        return Predicteddata,Targetdata, avg_test_loss, accuracy, log

def convert_to_edge_list(adj, device='cuda:3'):
        adj = torch.ones(20,20).triu(diagonal=1).to(device=device)
        Ixy = torch.nonzero(adj>0.1, as_tuple=False).t()
        edge_w = adj[Ixy[0], Ixy[1]]
        edge_list= Ixy
        # Return the edge list
        return edge_list, edge_w

def save_checkpoint(state, ct, comment = ''):
    # cwd = os.getcwd()
    # path = os.path.join(cwd,'Processed', comment + f'checkpoint{ct}.pth.tar')
    savelog(f'=> Not Saving checkpoint at {ct} for now', ct)
    # torch.save(state, path)


def load_model(path, model, input_size, hidden_size, output_size, future,num_layers, ct):
        savelog("Loading model from the checkpoint", ct)
        savelog('=> Loading checkpoint', ct)
        model = model(input_size, hidden_size, output_size, future, num_layers)
        criterion = nn.MSELoss()
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        return model, criterion

    

if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")