import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import os
from utilz.utils import *
import torch.nn.functional as F
import time

def train_model(model, optimizer, criterion, train_loader, test_loader, clip, args, datamax, datamin, ZoneConf):
    ct = args.ct
    epochs = args.epochs
    lr = args.learning_rate
    test_in_epoch = args.test_in_epoch
    patience_limit = args.patience_limit
    batch_size = args.batch_size
    columns_to_pick = args.Columns_to_Predict
    future = torch.tensor(args.future, device='cuda')
    input_size = args.input_size
    Nusers = args.Nusers
    sos = args.sos
    eos = args.eos
    n_heads = args.n_heads
    dssc = args.downsamplesc
    dstar = args.downsampletar
    prev_average_loss = 10000000.00000 #float('inf')  # Initialize with a high value
    epoch_losses = []
    test_loss = []
    scheduler = StepLR(optimizer, step_size=args.schd_stepzise, gamma=args.gamma)
    Max_Acc = 1000000 # collecting maximum accuracy so far
    patience = 0
    patienceLvl2 = 0
    device = args.device
    for epoch in range(epochs):
        patience += 1
        model.train()
        train_loss = []
        loss_num = 0
        
        for Scene, Target, Adj_Mat_Scene in train_loader:
            loss_num += 1 #Scene.size(0)
            optimizer.zero_grad()
            scene = Scene[:,0::dssc,:,:input_size].clone().detach()
            Target = Target[:,0::dstar,:,1:3].clone()
            scene = torch.cat((sos.repeat(scene.size(0),1,1,1), scene,eos.repeat(scene.size(0),1,1,1)), dim=1)
            target = torch.cat((sos[:,:,:,1:3].repeat(Target.size(0),1,1,1), Target,eos[:,:,:,1:3].repeat(Target.size(0),1,1,1)), dim=1).clone().detach().long()
            scene_mask = create_src_mask(scene)
            adj_mat = Adj_Mat_Scene[:,0::dssc].clone().detach()
            adj_zero = torch.ones_like(adj_mat[:,0]).unsqueeze(1)
            adj_mat = torch.cat((adj_zero, adj_mat, adj_zero), dim=1)

            scene_mask = scene_mask.permute(0,2,1)
            src_mask = scene_mask.reshape(-1, scene.size(1))
            # target_out = Targett[:, 0::2 ,:, 1:3]
            outputs = model(scene, src_mask,
                            adj_mat,target, 15) # 87*torch.ones_like(target[:,-1:,:,:2])*flg



            loss = criterion(outputs.reshape(-1, 1024), target[:,1:].reshape(-1))

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss.append(float(loss.item()))

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
                predicteddata,Targetdata, l, acc, log = test_model(model,criterion, test_loader, columns_to_pick, future, Nusers, input_size, sos, eos, args)
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

    if not test_in_epoch:
        Best_Predicteddata, Targetdata, l, acc, log = test_model(Best_Model,criterion, test_loader, columns_to_pick, future, Nusers, input_size, sos, eos, args)
        # predicteddata = torch.cat((predicteddata, p), dim=0)     
        test_loss.append([l, epoch])
        savelog(log, ct)
    return Best_Model, Best_Predicteddata, Targetdata, test_loss, epoch_losses, Max_Acc


def test_model(model,criterion, test_loader, columns_to_pick, future, Nusers, input_size, sos, eos, args):
    # savelog("Starting testing phase", ct)
    avg_inf_time = 0
    model.eval()
    test_losses = []
    test_loss_num = 0
    ll = 0   
    true_loss = 0
    lvl2_loss = 0
    n_heads= args.n_heads
    device = args.device
    dssc = args.downsamplesc
    dstar = args.downsampletar
    with torch.no_grad():
 
        for Scene, Target, Adj_Mat_Scene in test_loader:
            scene = Scene[:,0::dssc,:,:input_size].clone()
            scene = torch.cat((sos.repeat(scene.size(0),1,1,1), scene,eos.repeat(scene.size(0),1,1,1)), dim=1).clone()
            # Target = torch.cat((sos.repeat(Target.size(0),1,1,1), Target,eos.repeat(Target.size(0),1,1,1)), dim=1).clone()
            Target = Target[:,0::dstar,:,1:3]
            Target = torch.cat((sos[:,:,:,1:3].repeat(Target.size(0),1,1,1), Target,eos[:,:,:,1:3].repeat(Target.size(0),1,1,1)), dim=1).clone()
            scene_mask = create_src_mask(scene)
            scene_mask = scene_mask.permute(0,2,1)
            src_mask = scene_mask.reshape(-1, scene.size(1))
            adj_mat = Adj_Mat_Scene[:,0::dssc].clone()
            adj_zero = torch.ones_like(adj_mat[:,0]).unsqueeze(1)
            adj_mat = torch.cat((adj_zero, adj_mat, adj_zero), dim=1)
            # target = Target[:,:,:, :2].clone()
            # trgt_mask = target_mask(target[:,:1,:,:2])
            # trgt_mask = trgt_mask.permute(0,4,1,2,3)
            # trgt_mask = trgt_mask.reshape(-1, trgt_mask.size(-1), trgt_mask.size(-1))
            # flg = torch.logical_or((target[:,-1:,:,:1] !=0), (target[:,-1:,:,1:2] != 0))
        
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            test_loss_num += 1 #Scene.size(0)
            ll += Scene.size(0)
            start_event.record()

            Pred_target = model(scene, src_mask, adj_mat, Target[:,:1], 15) # 87*torch.ones_like(target[:,-1:,:,:2])*flg

            end_event.record()
            torch.cuda.synchronize()
            Pred_target = Pred_target.softmax(dim=-1)
            firstpick = Pred_target.argmax(dim=-1)

            val, prob = torch.topk(Pred_target, k=6, dim=-1)
            Pred_target = (val*prob).sum(-1)/val.sum(-1)
            
            # xy = (top_indices*top_values).sum(-1)/top_values.sum(-1)
            # x = (target[:,-1,:,:1]+66)*6
            # y = (target[:,-1,:,1:]+82)*6
            # tar = torch.cat((x,y), dim=-1)
            # true_loss += criterion(Pred_target.reshape(-1, 1024), target.reshape(-1).long())
            # test_losses.append(tests_loss.item())
            true_loss += F.mse_loss(firstpick, Target[:,1:])
            # a = Pred_target[:,0].softmax(dim=-1)
            # top_values, top_indices = torch.topk(a, k = 5, dim=-1)
            # xy = (top_indices*top_values).sum(-1)/top_values.sum(-1)
            # x = xy[:,:,:1]/6 - 66
            # y = xy[:,:,1:]/6 - 82
            # xy = torch.cat((x,y), dim=-1)
            # true_loss += F.mse_loss(top_indices[:,:,:, 0], target[:, -1])
            lvl2_loss += F.mse_loss(Pred_target, Target[:,1:])
            # if Trloss< LossThreshold:
            #     PredZones = zonefinder(outputs, ZoneConf)
            #     counts, totalzone, BtN, nonzero,doublezone = Zone_compare(PredZones, Target[:,-1,:32,6], Scene[:,-1,:32,6],outputs)
            #     CorrectZones += counts
            #     totalZones += totalzone
            #     totallen += BtN
            #     totalnonzero += nonzero
            #     totaldoublezone += doublezone
            # a = Pred_target.softmax(dim=-1)
            # b =a.argmax(dim=-1).squeeze(1)
            # c = torch.zeros_like(target[:,-1])
            # for i, boxes in enumerate(b):
            #     for n, box in enumerate(boxes):
            #         c[i,n] = torch.tensor(args.boxes[int(box)], device= c.device)[0]

            # lvl2_loss += F.mse_loss(target[:,-1], c[0])
            try:
                Predicteddata = torch.cat((Predicteddata, Pred_target), dim=0)
                Targetdata = torch.cat((Targetdata, Target[:,1:-1,:, :2]), dim=0)
            except:
                Predicteddata = Pred_target
                Targetdata = Target[:,1:-1,:, :2]

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

    
class CustomLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_mse=1.0):
        super(CustomLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.weight_ce = weight_ce
        self.weight_mse = weight_mse

    def forward(self, output_cross, output_mse, target_cross, target_mse):

        # Cross-Entropy Loss for classification task
        ce_loss = self.cross_entropy_loss(output_cross, target_cross)
        
        # MSE Loss for regression task
        mse_loss = self.mse_loss(output_mse, target_mse)
        
        # Weighted combination of the losses
        total_loss = self.weight_ce * ce_loss + self.weight_mse * mse_loss
        return total_loss
if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")