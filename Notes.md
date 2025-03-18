### LR1000_HS256_NL2_BS256 11-08-15-21 with Max Acc of 84.1084099870924 # : Max accuracy on the test result but higher test loss than a previous one

# Tips:
    * LR between 1e-2~1e-3 is the best
    * HS More than 256 is not prefered, 128 is optimal

Things to Try:

1: Decreased Hidden Size to 32: Higher Test loss but still works. Inference Time 18ms
2: Different LRs: LR3000_HS128_NL2_BS256 11-08-18-09 with Max Acc of 71.8988135340381, still had capacity for improvement
3: Scoial Pool Threshold
4: e = e*adj_mat threshold: didn't work, but e= e*adj_mat**2 seems to work: Saving result of LR1000_HS128_NL2_BS256 11-08-20-15 with Max Acc of 76.01486057119199
5: nn.Conv2d(20, 20, kernel_size=n_nodes, stride=1, padding=0, bias=False) worked the best
    LR1000_HS128_NL2_BS256 11-08-20-31 with Max Acc of 82.51238418159002
5: Add two more FC layers at the last stage
6: Added 'Vx', 'Vy', 'heading',xc,yc, Rc, SinX,CosX,SinY,CosY to enhance the accuracy: it worked, better than anything else! Saving result of LR1000_HS128_NL2_BS256 11-11-17-24 with Max Acc of 79.57686191847057
7: Let's predict xc, yc instead of x,y: not very effective, no changes
8: Added Sin2x,2y, Cos2x,2y and it converged much much faster
9: Changed Adj_Matrix from pow(-0.2).tanh() to pow(-2).tanh()
9.5: Use e*adj_mat
10: Let's change Adj_Mat Invsqr_dist = torch.sum(diff**2, dim=3).sqrt()
                        Invsqr_dist = torch.exp(-Invsqr_dist/1024) : Saving result of LR1000_HS128_NL2_BS128 11-12-19-08 with Max Acc of 79.09371488601175

11: Used BBx, BBy again, not a significant change around 1% increase
12: Best configuration: learning_rate, schd_stepzise, gamma, epochs, batch_size, patience_limit, clip= 1e-2, 30, 0.4, 400, 128, 40, 1

13: Let's run the minimal version! Ooooooooof, cts after 14-59-37: Really high inference time! much easier convergence, but high loss on test data
    With two normalization layer, it worked. Saving result of LR50_HS128_NL2_BS128 11-13-17-29 with Max Acc of 10.247368217305022

14: learning_rate, schd_stepzise, gamma, epochs, batch_size, patience_limit, clip= 2e-4, 20, 0.5, 300, 128, 40, 1 , ct = 11-13-20-39 
got more than 94.5% accuracy
Saving result of LR20_HS128_NL2_BS128 11-13-20-39 with Max Acc of 94.69413872781809

15: Had to change social pool to dissolve time stamp from it, Solution avg over time sequence
    self.socialpool = nn.Conv1d(n_nodes,n_nodes, kernel_size=n_nodes, stride=1, padding=0)
    threshold = self.socialpool(adj_mat.mean(dim = 1))
    threshold = adj_mat < threshold.unsqueeze(1)
    Saving result of LR20_HS128_NL2_BS128 11-14-17-11 with Max Acc of 97.58486598527816

16: Not a correct stacked GRU, now fixed it
------------------------------

17: ASTGCN Model is used,  LR500_HS128_NL1_BS1024 11-26-18-49 with Max Acc of 0.009040287014987278
Saving result of LR500_HS128_NL1_BS1024 11-26-18-49 with Max Acc of 0.009040287014987278

