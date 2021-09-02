import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
from util import count_parameters
from torch import nn
from util import print_loss, print_loss_sensor, print_loss_seq, print_loss_sensor_seq
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA', help='data path')
parser.add_argument('--suffix',type=str,default='_filtered_we', help='data file suffix')
parser.add_argument('--eRec',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--retrain',action='store_true',help='whether to load a previous model')
parser.add_argument('--checkpoint',type=str,help='')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='output sequence length')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='results/model',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()




def main():
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    # suffix = '_filtered_we'  # _filtered_we, _filtered_ew
    eR_seq_size = 24  # 24
    error_size = 6
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size,
                                   eRec=args.eRec, eR_seq_size=eR_seq_size, suffix=args.suffix)
    scaler = dataloader['scaler']

    blocks = int(dataloader[f'x_train{args.suffix}'].shape[1] / 3)  # Every block reduce the input sequence size by 3.
    print(f'blocks = {blocks}')

    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit, blocks, eRec=args.eRec, retrain=args.retrain, checkpoint=args.checkpoint,
                     error_size=error_size)

    if args.retrain:
        dataloader['val_loader'] = dataloader['train_loader']

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            if args.eRec:
                trainx = trainx.transpose(0, 1)
                trainy = trainy.transpose(0, 1)
            trainx= trainx.transpose(-3, -1)
            trainy = trainy.transpose(-3, -1)
            # print(f'trainx.shape = {trainx.shape}')
            # print(f'trainy.shape = {trainy.shape}')
            # print(f'trainy.shape final = {trainy[:,0,:,:].shape}')
            if args.eRec:
                metrics = engine.train(trainx, trainy[:, :, 0, :, :])
            else:
                metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            if args.eRec:
                testx = testx.transpose(0, 1)
                testy = testy.transpose(0, 1)
            testx = testx.transpose(-3, -1)
            testy = testy.transpose(-3, -1)
            if args.eRec:
                metrics = engine.eval(testx, testy[:, :, 0, :, :])
            else:
                metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = 82  # 24 hay que sumarle 1 para obtener el ID del modelo
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
    # engine.model.load_state_dict(torch.load(args.save + f"_id_25_2.6_best_model.pth"))
    # engine.model.load_state_dict(torch.load(args.save + f"_exp1_best_2.6.pth"))

    #torch.save(engine.model.state_dict(), args.save + f"_id_{bestid+1}_best_model.pth")
    print(f'best_id = {bestid+1}')

    outputs = []
    realy = torch.Tensor(dataloader[f'y_test{args.suffix}']).to(device)
    #print(f'realy: {realy.shape}')
    if args.eRec:
        realy = realy.transpose(0, 1)
        realy = realy.transpose(-3, -1)[-1,:,0,:,:]
        #print(f'realy2: {realy.shape}')
    else:
        realy = realy.transpose(-3, -1)[:, 0, :, :]
        #print(f'realy2: {realy.shape}')
    criterion = nn.MSELoss(reduction='none')  # L2 Norm
    criterion2 = nn.L1Loss(reduction='none')
    loss_mse_list = []
    loss_mae_list = []

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        if args.eRec:
            testx = testx.transpose(0, 1)
            testy = testy.transpose(0, 1)
        testx = testx.transpose(-3, -1)
        testy = testy.transpose(-3, -1)
        with torch.no_grad():
            if args.eRec:
                preds = engine.model(testx, testy[:, :, 0:1, :, :], scaler).transpose(1,3)
            else:
                preds = engine.model(testx).transpose(1,3)

        #print(f'preds: {scaler.inverse_transform(torch.squeeze(preds.transpose(-3, -1))).shape}')
        #print(f'testy: {torch.squeeze(testy[:, 0:1, :, :].transpose(-3, -1)).shape}')
        if args.eRec:
            loss_mse = criterion(scaler.inverse_transform(torch.squeeze(preds.transpose(-3, -1))), torch.squeeze(testy[-1, :, 0:1, :, :].transpose(-3, -1)))
            loss_mae = criterion2(scaler.inverse_transform(torch.squeeze(preds.transpose(-3, -1))), torch.squeeze(testy[-1, :, 0:1, :, :].transpose(-3, -1)))
        else:
            loss_mse = criterion(scaler.inverse_transform(torch.squeeze(preds.transpose(-3, -1))),
                                 torch.squeeze(testy[:, 0:1, :, :].transpose(-3, -1)))
            loss_mae = criterion2(scaler.inverse_transform(torch.squeeze(preds.transpose(-3, -1))),
                                  torch.squeeze(testy[:, 0:1, :, :].transpose(-3, -1)))

        loss_mse_list.append(loss_mse)
        loss_mae_list.append(loss_mae)

        outputs.append(preds.squeeze())

    loss_mse_list.pop(-1)
    loss_mae_list.pop(-1)
    loss_mse = torch.cat(loss_mse_list, 0)
    loss_mae = torch.cat(loss_mae_list, 0)
    #loss_mse = torch.squeeze(loss_mse).cpu()
    #loss_mae = torch.squeeze(loss_mae).cpu()
    loss_mse = loss_mse.cpu()
    loss_mae = loss_mae.cpu()
    print(f'loss_mae: {loss_mae.shape}')
    print(f'loss_mse: {loss_mae.shape}')

    res_folder = 'results/'
    original_stdout = sys.stdout
    with open(res_folder + f'loss_evaluation.txt', 'w') as filehandle:
        sys.stdout = filehandle  # Change the standard output to the file we created.
        count_parameters(engine.model)
        # loss_mae.shape --> (batch_size, seq_size, n_detect)
        print(' 1. ***********')
        print_loss('MSE', loss_mse)
        print(' 2. ***********')
        print_loss('MAE', loss_mae)
        print(' 3. ***********')
        print_loss_sensor('MAE', loss_mae)
        print(' 5. ***********')
        print_loss_seq('MAE', loss_mae)
        print(' 6. ***********')
        print_loss_sensor_seq('MAE', loss_mae)

        sys.stdout = original_stdout  # Reset the standard output to its original value

    with open(res_folder + f'loss_evaluation.txt', 'r') as filehandle:
        print(filehandle.read())


    yhat = torch.cat(outputs,dim=0)
    #print(f'yhat: {yhat.shape}')
    yhat = yhat[:realy.size(0),...]
    #print(f'yhat2: {yhat.shape}')


    print("Training finished")
    #print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over {:.4f} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(args.seq_length,np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(np.min(his_loss),2))+".pth")



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
